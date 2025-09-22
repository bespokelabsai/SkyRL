#!/bin/bash
# run_vllm.sh

MODEL="Qwen/Qwen3-32B"
export CUDA_VISIBLE_DEVICES=0

vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port 8001 \
  --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
  --max-model-len 131072 \
  --served-model-name "$MODEL" \
  --gpu-memory-utilization 0.95 \