import os
import tempfile
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

from modalities.config.config import ProcessGroupBackendType
from modalities.util import get_experiment_id_of_run
from tests.end2end_tests.custom_components import MultiProcessingCudaEnv


@pytest.fixture
def temporary_folder_path():
    with tempfile.TemporaryDirectory() as tmp_dir_path:
        yield Path(tmp_dir_path)


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="This test requires 2 GPUs.",
)
class TestExperimentIdGeneration:
    @staticmethod
    def _test_experiment_id_syncing_thread(
        process_id: int, world_size: int, rdvz_port: int, config_file_path: Path, temporary_folder_path: Path
    ):
        TestExperimentIdGeneration._run_experiment_id_generation_test_in_dist_env(
            process_id=process_id,
            world_size=world_size,
            rdvz_port=rdvz_port,
            config_file_path=config_file_path,
            temporary_folder_path=temporary_folder_path,
        )

    @staticmethod
    def _run_experiment_id_generation_test_in_dist_env(
        process_id: int, world_size: int, rdvz_port: int, config_file_path: Path, temporary_folder_path: Path
    ):
        with MultiProcessingCudaEnv(
            process_group_backend=ProcessGroupBackendType.nccl,
            global_rank=process_id,
            local_rank=process_id,
            world_size=world_size,
            rdvz_port=rdvz_port,
        ):
            experiment_id = get_experiment_id_of_run(
                config_file_path=config_file_path,
                hash_length=8,
                max_experment_id_byte_length=1024,
            )
            # write experiment_id to file
            experiment_id_file_path = temporary_folder_path / f"{process_id}.txt"
            with open(experiment_id_file_path, "w") as f:
                f.write(experiment_id)

    @staticmethod
    @pytest.mark.parametrize(
        "rdvz_port, relative_config_path",
        [
            (22358, "../end2end_tests/system_tests/configs/fsdp2_gpt2_train_num_steps_8.yaml"),
        ],
    )
    def test_experiment_id_generation(rdvz_port, relative_config_path: str, temporary_folder_path: Path):
        working_dir = Path(os.path.dirname(__file__))
        config_file_path = working_dir / relative_config_path
        world_size = 2
        mp.spawn(
            TestExperimentIdGeneration._test_experiment_id_syncing_thread,
            args=(world_size, rdvz_port, config_file_path, temporary_folder_path),
            nprocs=world_size,
            join=True,
        )
        # Check if all experiment_id files are the same
        experiment_id_files = [temporary_folder_path / f"{i}.txt" for i in range(world_size)]
        experiment_ids = []
        for experiment_id_file in experiment_id_files:
            with open(experiment_id_file, "r") as f:
                experiment_ids.append(f.read().strip())
        assert len(experiment_ids) == world_size, "Not all processes have generated an experiment ID file"
        assert len(set(experiment_ids)) == 1, "Experiment IDs are not the same across processes"
