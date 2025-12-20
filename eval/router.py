from ray import serve
from ray.serve.llm import LLMConfig, build_openai_app


llm_config1 = LLMConfig(
    model_loading_config=dict(
        model_id="bespokelabs/Qwen3-8B-ot_step60_high",
        model_source="bespokelabs/Qwen3-8B-ot_step60_high",
    ),
    deployment_config=dict(
        autoscaling_config=dict(
            min_replicas=8, max_replicas=8,
        )
    ),
    accelerator_type="H100",
)

app = build_openai_app({"llm_configs": [llm_config1]})
serve.run(app, blocking=True)