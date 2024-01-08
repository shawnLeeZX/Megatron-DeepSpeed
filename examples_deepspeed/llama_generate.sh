#!/bin/bash
# This example script is contributed by external user https://github.com/LydiaXiaohongLi
set -ex

######################################
# Change the below configurations here
BASE_PATH=./tmp
DS_CONFIG=${BASE_PATH}/deepspeed.json
LOAD_CHECKPOINT_PATH=/root/models/llama-7b-megads
TOKENIZER_PATH=/root/models/llama-7b-megads # offical llama tokenizer.model

TP=2
PP=2
ZERO_STAGE=0

GPUS_PER_NODE=2
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=2
NODE_RANK=0

HIDDEN_SIZE=4096 # e.g. llama-13b: 5120
FFN_HIDDEN_SIZE=11008 # e.g. llama-13b: 13824
NUM_LAYERS=32
NUM_HEADS=32 # e.g. llama-13b: 40
SEQ_LENGTH=2048
MAX_NEW_TOKENS=30

MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=3 # e.g. llama: 4M tokens
TRAIN_STEPS=1000 # e.g. llama: 1T tokens / 4M tokens_per_batch = 250000 steps
LR=3e-4
MIN_LR=3e-5
LR_WARMUP_STEPS=100
WEIGHT_DECAY=0.1
GRAD_CLIP=1

## Activation checkpointing saves GPU memory, but reduces training speed
# activation_checkpoint="true"
activation_checkpoint="false"

# Below configuration required for llama model as per llama paper
# --no-query-key-layer-scaling \
# --attention-dropout 0 \
# --hidden-dropout 0 \
# --use-rotary-position-embeddings \
# --untie-embeddings-and-output-weights \
# --swiglu \
# --normalization rmsnorm \
# --disable-bias-linear \
######################################



cat <<EOT > $DS_CONFIG
{
  "train_batch_size" : $GLOBAL_BATCH_SIZE,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "steps_per_print": 1,
  "zero_optimization": {
    "stage": $ZERO_STAGE
  },
  "bf16": {
    "enabled": true
  }
}
EOT

ds_args=""
ds_args=" --deepspeed ${ds_args}"
ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"
ds_args=" --zero-stage=$ZERO_STAGE ${ds_args}"

if [ "${activation_checkpoint}" = "true" ]; then
  ds_args="--deepspeed-activation-checkpointing ${ds_args}"

  ## old argument for recomputing the transformer layer
  # ds_args="--checkpoint-activations ${ds_args}"

  ## new argument for recomputing the transformer layer
  ds_args="--recompute-granularity full --recompute-method uniform ${ds_args}"
  ## new argument for recomputing only the attention layer
  # ds_args="--recompute-granularity selective ${ds_args}"
fi

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

# torchrun $DISTRIBUTED_ARGS \
deepspeed --num_nodes=2 --num_gpus=2 \
      --hostfile=/root/paddlejob/workspace/hostfile \
      --master_port "${MASTER_PORT}" \
      megatron/ds_text_generation.py \
      --tensor-model-parallel-size $TP \
      --pipeline-model-parallel-size $PP \
      --num-layers $NUM_LAYERS \
      --hidden-size $HIDDEN_SIZE \
      --ffn-hidden-size $FFN_HIDDEN_SIZE \
      --num-attention-heads $NUM_HEADS \
      --micro-batch-size $MICRO_BATCH_SIZE \
      --global-batch-size $GLOBAL_BATCH_SIZE \
      --seq-length $SEQ_LENGTH \
      --max-new-tokens $MAX_NEW_TOKENS \
      --max-position-embeddings $SEQ_LENGTH \
      --finetune \
      --load $LOAD_CHECKPOINT_PATH \
      --data-path $DATASET \
      --data-impl mmap \
      --temperature 1.0 \
      --top_k 1 \
      --top_p 0.0 \
      --tokenizer-type HFTokenizer \
      --distributed-backend nccl \
      --log-interval 1 \
      --save-interval 10000 \
      --eval-interval 1000 \
      --eval-iters 10 \
      --bf16 \
      --no-query-key-layer-scaling \
      --attention-dropout 0 \
      --hidden-dropout 0 \
      --use-rotary-position-embeddings \
      --untie-embeddings-and-output-weights \
      --swiglu \
      --tensorboard-dir ./tmp/tensorboard \
      --normalization rmsnorm \
      --disable-bias-linear \
      --tokenizer-model $TOKENIZER_PATH \
      $ds_args


      # --out-seq-length $SEQ_LENGTH \