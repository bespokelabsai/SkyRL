set -x

# WORK IN PROGRESS
# Colocated GRPO training+generation for Qwen2.5-1.5B-Instruct on TerminalBench tasks.

# uv run examples/terminal_bench/prepare_dataset.py --task_dir $HOME/data/terminal_bench/tasks --output_dir $HOME/data/terminal_bench --output_name train
# export WANDB_API_KEY=<your_key_here>
# bash examples/terminal_bench/run_tbench_gen.sh
# /root/ckpts/ez_sweagent_8B_ckpt/global_step_13/policy
DATA_DIR="$HOME/data/terminal_bench"
NUM_GPUS=4
LOGGER="wandb"  # change to "wandb" to export to wandb
TBENCH_CONFIG_DIR="examples/terminal_bench"
SANDBOXES_DIR="sandboxes" # TODO: For now, `sandboxes` is cloned into SkyRL/skyrl-train.

uv run --isolated --extra vllm --extra sandboxes --with "sandbox@./sandboxes" -m examples.terminal_bench.entrypoints.main_tbench_generate \
  data.train_data="['$DATA_DIR/train.parquet']" \
  hydra.searchpath=[file://$TBENCH_CONFIG_DIR] \
  +terminal_bench_config=terminal_bench \
  terminal_bench_config.max_episodes=32 \
  terminal_bench_config.sandboxes_dir=$SANDBOXES_DIR \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="Qwen/Qwen3-8B" \
  trainer.placement.colocate_all=true \
  trainer.strategy=deepspeed \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=2 \
  generator.inference_engine_tensor_parallel_size=2 \
  trainer.epochs=5 \
  trainer.eval_batch_size=4 \
  trainer.eval_before_train=false \
  trainer.eval_interval=-1 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=8 \
  trainer.policy_mini_batch_size=4 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=1000000 \
  trainer.max_prompt_length=16000 \
  generator.enable_http_endpoint=true \
  generator.http_endpoint_host="127.0.0.1" \
  generator.http_endpoint_port=8000 \
  generator.sampling_params.max_generate_length=16000 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=true \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=4 \
  generator.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="terminal_bench" \
  trainer.run_name="terminal_bench_test" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$HOME/ckpts/ez_sweagent_8B_ckpt" \
  trainer.gradient_checkpointing_use_reentrant=true \
  trainer.ref.fsdp_config.cpu_offload=false
  $@