rm megatron-deepspeed.tar.gz
tar --exclude="output" --exclude="core.*" -cvzf megatron-deepspeed.tar.gz .
bash cp.sh megatron-deepspeed.tar.gz
mpirun tar -xf $HOME/Megatron-DeepSpeed/megatron-deepspeed.tar.gz -C $HOME/Megatron-DeepSpeed
