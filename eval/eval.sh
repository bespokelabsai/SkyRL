MODEL="bespokelabs/Qwen3-8B-ot_step100"
JOB_NAME="100step_otsetup"

export DAYTONA_API_KEY="dtn_19b26f08b6a7f295e341c23e64096af54fb2b008e38415c4221f2264f17ff791"

harbor jobs start \
  --dataset "terminal-bench@2.0" \
  --n-concurrent 8 \
  --agent terminus-2 \
  --model "hosted_vllm/$MODEL" \
  --env "daytona" \
  --agent-kwarg "api_base=http://localhost:8000/v1" \
  --agent-kwarg "key=fake_key" \
  --agent-kwarg "max_tokens=16384" \
  --agent-kwarg "model_info={\"max_output_tokens\":16384,\"max_input_tokens\":32768}" \
  --n-attempts 3 \
  --job-name "$JOB_NAME" \
  --config "config.yaml"