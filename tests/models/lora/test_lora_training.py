import os
from pathlib import Path
from tabnanny import check

import pytest
import torch

from modalities.__main__ import Main
from modalities.config.config import ProcessGroupBackendType
from modalities.config.instantiation_models import TrainingComponentsInstantiationModel
from modalities.running_env.cuda_env import CudaEnv
from tests.conftest import _ROOT_DIR


# NOTE: We need to run the tests in a torch distributed environment with at least two GPUs.
# CUDA_VISIBLE_DEVICES=0,1 torchrun --rdzv-endpoint localhost:29502 --nnodes 1 --nproc_per_node 2 \
#   $(which pytest) path/to/test_lora_training.py


@pytest.fixture()
def config_file_name() -> str:
    return "lora_training.yaml"


@pytest.fixture()
def config_file_path(config_file_name) -> Path:
    config_file_path = _ROOT_DIR / Path("tests/models/lora/test_configs/" + config_file_name)
    if not os.path.exists(config_file_path):
        raise FileNotFoundError("Config file doesn't exist")
    return config_file_path


@pytest.fixture
def checkpointing_path(tmp_path):
    return tmp_path.parent


@pytest.fixture
def main_obj(config_file_path, checkpointing_path):
    print(checkpointing_path)
    main_obj = Main(config_file_path)
    path_as_str = str(checkpointing_path)
    main_obj.config_dict["settings"]["paths"]["checkpoint_saving_path"] = path_as_str
    main_obj.config_dict["checkpoint_saving"]["config"]["checkpoint_saving_execution"]["config"][
        "checkpoint_path"
    ] = path_as_str
    return main_obj


@pytest.mark.skipif(
    "RANK" not in os.environ or torch.cuda.device_count() < 2,
    reason="This e2e test requires 2 GPU and a torchrun distributed environment.",
)
def test_lora_model_training_on_one_gpu(main_obj, checkpointing_path):
    with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
        components = main_obj.build_components(components_model_type=TrainingComponentsInstantiationModel)
        main_obj.run(components)

    assert os.path.exists(checkpointing_path)

    checkpoint_files = []
    for root, dirs, files in os.walk(checkpointing_path):
        for file in files:
            if "model" in file or "optimizer" in file or file.endswith(".yaml"):
                checkpoint_files.append(file)
    if torch.cuda.current_device() == 0:
        assert len(checkpoint_files) >= 3, "Output of the test i.e. a model checkpoint was not created!"
