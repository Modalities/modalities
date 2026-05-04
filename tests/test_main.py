import tempfile
from pathlib import Path

import pytest
import torch.cuda

from modalities.__main__ import Main
from modalities.config.config import ProcessGroupBackendType
from modalities.config.instantiation_models import TrainingComponentsInstantiationModel
from modalities.running_env.cuda_env import CudaEnv


@pytest.fixture
def temporary_folder_path():
    with tempfile.TemporaryDirectory() as tmp_dir_path:
        yield Path(tmp_dir_path)


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="This e2e test requires 1 GPU.")
def test_e2e_training_run_wout_ckpt(monkey_patch_dist_env, dummy_config_path, temporary_folder_path: Path):
    with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
        main = Main(dummy_config_path, experiments_root_path=temporary_folder_path)
        # main.config_dict = dummy_config
        components = main.build_components(components_model_type=TrainingComponentsInstantiationModel)
        main.run(components)
