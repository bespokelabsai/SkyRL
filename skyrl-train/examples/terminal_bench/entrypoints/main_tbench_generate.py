"""
Main entrypoint for generating rollouts on terminal bench tasks.
"""

from importlib import import_module
import ray
import asyncio
import hydra
from loguru import logger
from omegaconf import DictConfig

from skyrl_train.utils import validate_cfg
from skyrl_train.utils.utils import initialize_ray
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.entrypoints.main_base import (
    create_ray_wrapped_inference_engines_from_config,
    create_remote_inference_engines_from_config,
    BasePPOExp,
    config_dir,
)
from skyrl_train.generators.base import GeneratorInput
from examples.terminal_bench.generator.terminal_bench_generator import TerminalBenchGenerator


class TerminalBenchGenerateExp(BasePPOExp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        """
        Initializes the TerminalBenchGenerator.
        """
        return TerminalBenchGenerator(
            generator_cfg=cfg.generator,
            terminal_bench_cfg=cfg.terminal_bench_config,  # Pass terminal_bench config to the generator
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
        )

    def _setup_generator(self):
        logger.info(self.get_cfg_as_str(self.cfg))

        tokenizer = self.tokenizer
        if self.cfg.generator.run_engines_locally:
            inference_engines = create_ray_wrapped_inference_engines_from_config(self.cfg, self.colocate_pg, tokenizer)
        else:
            inference_engines = create_remote_inference_engines_from_config(self.cfg, tokenizer)

        inference_engine_client = InferenceEngineClient(inference_engines, tokenizer, self.cfg)
        asyncio.run(inference_engine_client.wake_up())

        return self.get_generator(self.cfg, tokenizer, inference_engine_client)

    def run(self):
        generator = self._setup_generator()

        # Build input from the training dataset
        num_prompts = len(self.train_dataset)
        prompts = []
        env_extras = []
        for i in range(num_prompts):
            prompt, _, extra = self.train_dataset[i]
            prompts.append(prompt)
            env_extras.append(extra)

        input_batch = GeneratorInput(
            prompts=prompts,
            env_classes=None,
            env_extras=env_extras,
            sampling_params=None,
        )

        # Start generation
        generator_output = asyncio.run(generator.generate(input_batch))
        # import pdb; pdb.set_trace()
        reward = generator_output['rewards']
        print(reward)
        accuracy = sum(reward) / len(reward)
        print(f"Accuracy: {accuracy:.4f}")


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):
    # make sure that the training loop is not run on the head node.
    exp = TerminalBenchGenerateExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    # validate the arguments
    validate_cfg(cfg)

    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
