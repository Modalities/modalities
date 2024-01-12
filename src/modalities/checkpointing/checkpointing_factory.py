from modalities.checkpointing.checkpointing import (
    Checkpointing,
    CheckpointingExecutionIF,
    CheckpointingIF,
    CheckpointingStrategyIF,
)
from modalities.config.config import CheckpointingConfig
from modalities.resolver_register import ResolverRegister
from modalities.running_env.fsdp.fsdp_running_env import RunningEnv


class CheckpointingFactory:
    @staticmethod
    def get_checkpointing(
        resolvers: ResolverRegister,
        config: CheckpointingConfig,
        running_env: RunningEnv,
        experiment_id: str,
        num_ranks: int,
    ) -> CheckpointingIF:
        checkpointing_strategy: CheckpointingStrategyIF = resolvers.build_component_by_config(
            config=config.checkpointing_strategy, extra_kwargs={}
        )

        checkpointing_execution: CheckpointingExecutionIF = resolvers.build_component_by_config(
            config=config.checkpointing_execution,
            extra_kwargs={"experiment_id": experiment_id, "model_wrapping_fn": running_env.wrap_model},
        )

        checkpointing = Checkpointing(
            checkpointing_strategy=checkpointing_strategy,
            checkpointing_execution=checkpointing_execution,
            num_ranks=num_ranks,
        )

        return checkpointing
