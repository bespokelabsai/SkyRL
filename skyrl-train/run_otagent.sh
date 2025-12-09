# My key
export DAYTONA_API_KEY="dtn_19b26f08b6a7f295e341c23e64096af54fb2b008e38415c4221f2264f17ff791"
export WANDB_API_KEY="854a2b39e99ffee11c76d1003eb8a777045687e9"

# Got after hf download open-thoughts/OpenThoughts-Agent-v1-RL --repo-type=dataset
# cd into the downloaded folder, say /path/to/.cache/huggingface/hub/datasets--open-thoughts--OpenThoughts-Agent-v1-RL/snapshots/hash_code
# python extract_parquet_tasks.py tasks_new.parquet ./extracted_tasks
TRAIN_DATA="['$HOME/ez_apex_281']"
# Got after hf download open-thoughts/OpenThoughts-TB-dev --repo-type=dataset
EVAL_DATA="['$HOME/.cache/huggingface/hub/datasets--open-thoughts--OpenThoughts-TB-dev/snapshots/0d54f719f34dca712c8d6ef0f51df4670a2a287a']"

CHAT_TEMPLATE_PATH="$HOME/SkyRL/skyrl-train/examples/terminal_bench/qwen3_thinking_acc.jinja2"
TRIALS_DIR="$HOME/trials"
CKPTS_DIR="$HOME/ckpts"
EXPORTS_DIR="$HOME/ckpts_hf"

# Run SkyRL command
python -m examples.terminal_bench.entrypoints.main_tbench \
  data.train_data=$TRAIN_DATA \
  data.val_data=$EVAL_DATA \
  trainer.policy.model.path=open-thoughts/OpenThinker-Agent-v1-SFT \
  hydra.searchpath=['file://examples/terminal_bench'] \
  +terminal_bench_config=terminal_bench \
  +terminal_bench_config.agent_name=terminus \
  +terminal_bench_config.max_episodes=64 \
  +terminal_bench_config.trials_dir=$TRIALS_DIR \
  +terminal_bench_config.override_memory_mb=2048 \
  +terminal_bench_config.override_storage_mb=4096 \
  +terminal_bench_config.override_cpus=2 \
  +terminal_bench_config.enable_summarize=false \
  trainer.export_path=$EXPORTS_DIR \
  trainer.ckpt_path=$CKPTS_DIR \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_nodes=1 \
  trainer.placement.ref_num_nodes=1 \
  trainer.placement.policy_num_gpus_per_node=8 \
  trainer.placement.ref_num_gpus_per_node=8 \
  generator.num_inference_engines=8 \
  generator.inference_engine_tensor_parallel_size=1 \
  +generator.engine_init_kwargs.custom_chat_template_chat_completion_path=$CHAT_TEMPLATE_PATH \
  trainer.epochs=3 \
  trainer.eval_batch_size=128 \
  trainer.eval_before_train=false \
  trainer.eval_interval=20 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=64 \
  trainer.policy_mini_batch_size=64 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=2 \
  trainer.hf_save_interval=2 \
  trainer.max_prompt_length=2048 \
  generator.sampling_params.max_generate_length=30720 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.n_samples_per_prompt=8 \
  generator.eval_n_samples_per_prompt=8 \
  generator.gpu_memory_utilization=0.8 \
  trainer.logger=wandb \
  trainer.project_name=apex_rl \
  trainer.run_name=ot8b_resumed \
  trainer.resume_mode=latest \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=false \
  generator.enable_http_endpoint=true \
  generator.http_endpoint_host=127.0.0.1 \
  generator.http_endpoint_port=8000
