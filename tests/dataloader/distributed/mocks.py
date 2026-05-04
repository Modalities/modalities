import os


class MultiProcessingCudaEnvMock:
    """Context manager to set the CUDA environment for distributed training."""

    def __init__(
        self,
        global_rank: int,
        local_rank: int,
        world_size: int,
        rdvz_port: int,
    ) -> None:
        self.global_rank = global_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.rdvz_port = rdvz_port
        self._original_env: dict[str, str | None] = {}

    def __enter__(self):
        # Store original values
        for key in ["MASTER_ADDR", "MASTER_PORT", "RANK", "LOCAL_RANK", "WORLD_SIZE"]:
            self._original_env[key] = os.environ.get(key)

        # Set new environment variables
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(self.rdvz_port)
        os.environ["RANK"] = str(self.global_rank)
        os.environ["LOCAL_RANK"] = str(self.local_rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)

        # torch.cuda.set_device(local_rank)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original environment variables
        for key, value in self._original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
