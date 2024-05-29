import math

from pydantic import BaseModel


class LocalNumBatchesConfig(BaseModel):
    num_ranks: int
    local_micro_batch_size: int
    global_num_samples: int


class NumberConversion:
    @staticmethod
    def get_local_num_batches(num_ranks: int, local_micro_batch_size: int, global_num_samples: int) -> int:
        """Calculates the number of local batches for each rank, given the global
        number of samples, local micro batch size and number of ranks.
        This helper function is primarily used to calculate the number of batches to
        skip when resuming a dataloader during warmstart.

        Args:
            num_ranks (int): _description_
            local_micro_batch_size (int): _description_
            global_num_samples (int): _description_

        Returns:
            int: _description_
        """
        return math.floor(global_num_samples / (num_ranks * local_micro_batch_size))
