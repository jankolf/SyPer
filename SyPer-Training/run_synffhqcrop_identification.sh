#!/bin/bash
export OMP_NUM_THREADS=4

# $1 = .py file, e.g. distributed_train_quantization_synthetic.py
# $2 = Config file path, generated using configs/generate_*.py
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 \
    --node_rank=0 --master_addr="127.0.0.1" --master_port=26001 $1 $2
