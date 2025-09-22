#!/bin/bash
# run_sb.sh

DATASET_PATH="/root/sandboxes-tasks"
MODEL="hosted_vllm/Qwen/Qwen2.5-Coder-32B-Instruct"

sb run \
  --dataset-path "$DATASET_PATH" \
  --n-concurrent 5 \
  --agent terminus-2 \
  --model "$MODEL" \
  --env docker \
  --agent-kwarg "api_base=http://localhost:8000/v1" \
  


  # --agent-kwarg "max_episodes=64"
