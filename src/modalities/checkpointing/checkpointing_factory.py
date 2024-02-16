from modalities.checkpointing.checkpointing import CheckpointingExecutionIF
from modalities.checkpointing.checkpointing_execution import FSDPToDiscCheckpointing
from modalities.running_env.fsdp.fsdp_running_env import RunningEnv


class CheckpointingExecutionFactory:
    @staticmethod
    def get_fsdp_to_disc_checkpointing(
        checkpoint_path: str, global_rank: int, running_env: RunningEnv, experiment_id: str
    ) -> CheckpointingExecutionIF:
        fsdp_to_disc_checkpointing = FSDPToDiscCheckpointing(
            checkpoint_path=checkpoint_path,
            global_rank=global_rank,
            experiment_id=experiment_id,
            model_wrapping_fn=running_env.wrap_model,
        )
        return fsdp_to_disc_checkpointing
