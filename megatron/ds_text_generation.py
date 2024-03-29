# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Pretrain utilities."""

from datetime import datetime
import math
import sys
import time
import json
from functools import partial
# The earliest we can measure the start time.
_TRAIN_START_TIME = time.time()
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.types import Number
from typing import Dict, List, Any
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron import (
    get_args,
    get_timers,
    get_tokenizer,
    print_rank_0, 
    is_rank_0
)
from megatron.utils import _expand_mask, _make_causal_mask, LeftPaddingCollator
from megatron.core import mpu, tensor_parallel
from megatron.model import Float16Module, GPTModel, GPTModelPipe
from megatron.core.enums import ModelType
from megatron.initialize import initialize_megatron
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.data.data_samplers import build_pretraining_data_loader
from megatron.utils import unwrap_model, update_rotary_pos_emb, set_backend_seq_length
from megatron.arguments import core_transformer_config_from_args
from megatron.text_generation.sampling import sample
from megatron.checkpointing import load_checkpoint
from megatron.core.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_src_rank,
)

import deepspeed
from deepspeed.runtime.utils import see_memory_usage
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.data_pipeline.data_routing.helper import convert_to_random_ltd
from megatron.model.transformer import ParallelTransformerLayer
from deepspeed import comm as dist

class InferenceParams:
    """Inference parameters that are passed to the main model in order
    to efficienly calculate and store the context during inference."""

    def __init__(self, max_batch_size, max_sequence_len):
        """Note that offsets are set to zero and we always set the
        flag to allocate memory. After the first call, make sure to
        set this flag to False."""
        self.max_sequence_len = max_sequence_len
        self.max_batch_size = max_batch_size
        self.sequence_len_offset = 0
        self.next_sequence_len = 0
        self.batch_size_offset = 0
        self.attn_mask = None
        self.first_forward = True
        self.key_value_memory_dict = {}

    def swap_key_value_dict(self, batch_idx):
        "swap between batches"
        if len(self.key_value_memory_dict) == 0:
            raise ValueError("should not swap when dict in empty")
        
        for layer_number in self.key_value_memory_dict.keys():
            inference_key_memory, inference_value_memory = self.key_value_memory_dict[layer_number]
            assert len(batch_idx) == inference_key_memory.shape[1] ## make sure batch size is the same
            new_inference_key_memory = inference_key_memory[:, batch_idx]
            new_inference_value_memory = inference_value_memory[:, batch_idx]
            self.key_value_memory_dict[layer_number] = (
                    new_inference_key_memory, new_inference_value_memory)
    
    def update(self):
        self.sequence_len_offset = self.next_sequence_len
        self.first_forward = False

    def set_attn_mask(self, attn_mask):
        """Set the attention mask."""
        self.attn_mask = attn_mask

def tensor_parallel_sample(logits, top_p=0.0, top_k=0, temperature=1.0):

    world_size = get_tensor_model_parallel_world_size()
    rank = get_tensor_model_parallel_rank()
    dst_rank = get_tensor_model_parallel_src_rank()

    if rank == 0:
        tensor_list = [torch.empty_like(logits) for _ in range(world_size)]
    else:
        tensor_list = []
    torch.distributed.gather(logits, tensor_list, dst=dst_rank, group=get_tensor_model_parallel_group())
    
    if rank == 0:
        logits = torch.cat(tensor_list, dim=-1).contiguous()
        new_token = sample(logits, top_p=top_p, top_k=top_k, temperature=temperature)
    else:
        new_token = torch.empty_like(logits[..., 0])

    return new_token

def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building GPT model ...')
    see_memory_usage(f"Before Building Model", force=True)

    args = get_args()
    tokenizer = get_tokenizer()
    config = core_transformer_config_from_args(args)
    with deepspeed.zero.Init(sequence_data_parallel_group=mpu.get_sequence_data_parallel_group(),
                             remote_device=None if args.remote_device == 'none' else args.remote_device,
                             config_dict_or_path=args.deepspeed_config,
                             enabled=args.zero_stage == 3,
                             mpu=mpu):
        
        if args.deepspeed and not args.no_pipeline_parallel:
            inference_params_cls = partial(InferenceParams, 
                                           max_batch_size=args.micro_batch_size,
                                           max_sequence_len=args.seq_length)
            
            sample_fn = partial(tensor_parallel_sample, 
                                temperature=args.temperature, 
                                top_k=args.top_k, 
                                top_p=args.top_p)

            model = GPTModelPipe(
                config=config,
                num_tokentypes=0,
                parallel_output=True,
                sample_fn=sample_fn,
                inference_params_cls=inference_params_cls,
                eos_id=tokenizer.eos,
                pad_id=tokenizer.pad
            )

            # This is a hack to give us a reference to get_batch_pipe from within training.py
            # We need to call model.set_batch_fn after deepspeed.initialize
            model._megatron_batch_fn = get_batch_pipe

            set_backend_seq_length(args.seq_length)
        else:
            assert False, "Only DeepSpeed model is supported"

    see_memory_usage(f"After Building Model", force=True)
    return model

def _create_ds_config_dict():
    args = get_args()
    if isinstance(args.deepspeed_config, dict) :
        ds_config_dict = args.deepspeed_config
    else:
        with open(args.deepspeed_config, 'r', encoding='utf-8') as config_file:
            ds_config_dict = json.load(config_file)

    if args.universal_checkpoint:
        ds_config_dict["checkpoint"] = {"load_universal": True}

    # Clear config path
    args.deepspeed_config = None 

    return ds_config_dict

def get_batch_pipe(data):
    """Modification of `get_batch` to work on `next(data_iterator)` instead of `data_iterator`"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['input_ids']
    datatype = torch.int64
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)
    tokens = data_b['input_ids'].long()

    keys = ['attention_mask']
    datatype = torch.bool
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)
    attention_mask = data_b['attention_mask'].bool()
    
    max_length = min(tokens.shape[-1] + args.max_new_tokens, args.seq_length)
    combined_attention_mask = _make_causal_mask(tokens.shape, max_length=args.seq_length) \
        .to(get_accelerator().current_device_name())
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    expanded_attn_mask = _expand_mask(attention_mask, max_length=args.seq_length).bool()
    combined_attention_mask = expanded_attn_mask + combined_attention_mask

    position_ids = torch.arange(tokens.shape[1], dtype=torch.long, device=get_accelerator().current_device_name())
    # suit for megatron pipe module format.
    return (tokens, position_ids, combined_attention_mask), ()

def setup_model(model_provider_func, model_type):
    """Setup model and optimizer."""
    args = get_args()

    model = get_model(model_provider_func, model_type)

    if args.deepspeed:
        args.deepspeed_config_dict = _create_ds_config_dict()
        print_rank_0("DeepSpeed is enabled.")
        model, *_ = deepspeed.initialize(
            model=model[0],
            args=args,
            mpu=mpu if args.no_pipeline_parallel else None,
            config=args.deepspeed_config_dict,
        )
        if isinstance(model, deepspeed.PipelineEngine):
            # hack to get batch_fn from pretrain_gpt.py
            model.set_batch_fn(model.module._megatron_batch_fn)

            assert model.grid.get_pipe_parallel_rank() == mpu.get_pipeline_model_parallel_rank()
            assert model.grid.get_slice_parallel_rank() == mpu.get_tensor_model_parallel_rank()
            assert model.grid.get_data_parallel_rank() == mpu.get_data_parallel_rank()
        model = [model]

    assert args.load
    timers = get_timers()
    timers('load-checkpoint', log_level=0).start(barrier=True)
    args.iteration = load_checkpoint(model, None, None)
    timers('load-checkpoint').stop(barrier=True)
    timers.log(['load-checkpoint'])

    # We only support local DDP with multiple micro-batches.
    if len(model) > 1 or mpu.get_pipeline_model_parallel_world_size() > 1:
        assert args.DDP_impl == 'local'

    # get model without FP16 and/or TorchDDP wrappers
    # if args.iteration == 0 and len(unwrapped_model) == 1 \
    #     and hasattr(unwrapped_model[0], 'init_state_dict_from_bert'):
    #     print_rank_0("Initializing ICT from pretrained BERT model")
    #     unwrapped_model[0].init_state_dict_from_bert()

    # random-LTD requires converting transformer layers
    if args.random_ltd:
        model[0] = convert_to_random_ltd(model[0], ParallelTransformerLayer)

    return model

def get_model(model_type=ModelType.encoder_or_decoder, wrap_with_ddp=True):
    """Build the model."""
    args = get_args()
    args.model_type = model_type

    # Build model.
    model = model_provider()
    model.model_type = model_type

    if not isinstance(model, list):
        model = [model]

    # Disallow training and inference with Transformer Engine
    # for non-GPT models
    args.allow_transformer_engine = all([type(m) == GPTModel for m in model])
    assert args.allow_transformer_engine or args.transformer_impl == 'local', \
        'Transformer Engine is only approved for GPT models'

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for model_module in model:
        for param in model_module.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    # Print number of parameters.
    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on (tensor, pipeline) '
              'model parallel rank ({}, {}): {}'.format(
            mpu.get_tensor_model_parallel_rank(),
            mpu.get_pipeline_model_parallel_rank(),
            sum([sum([p.ds_numel if hasattr(p,'ds_id') else p.nelement() for p in model_module.parameters()])
                 for model_module in model])), flush=True)

    return model

def main(args_defaults = None):
    initialize_megatron(args_defaults=args_defaults)
    args = get_args()
    tokenizer = get_tokenizer()
    data = [{"text":"This can be achiefved by directly using the LlamaTokenizer class, or passing in"},
            {"text":"Of cause, I'm not a fan of the new movie. It's too bad that"},
            {"text":"Of cause, I'm not a fan of the new movie. It's too bad that"},
            {"text":"the angry owner doused Tuffy with boiling hot water and threw him off a"}
            ]
    inputs_ids = list(map(lambda dict: {"input_ids": torch.tensor(tokenizer.tokenize(dict['text']))}, data))    
    collator = LeftPaddingCollator(tokenizer.pad)
    data_loader = build_pretraining_data_loader(inputs_ids, args.consumed_train_samples, collate_fn=collator)
    data_iter = iter(data_loader)
    model = setup_model(model_provider_func=model_provider, model_type=ModelType.encoder_or_decoder)
    result = model[0].generate_batch(data_iter, max_new_tokens=args.max_new_tokens)
    if dist.get_rank() == 0:
        print(result)

if __name__ == "__main__":
    main(args_defaults={'tokenizer_type': 'HFTokenizer'})