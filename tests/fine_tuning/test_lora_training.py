import os
from pathlib import Path

import pytest
import torch

from modalities.__main__ import Main
from modalities.config.config import ProcessGroupBackendType
from modalities.config.instantiation_models import TrainingComponentsInstantiationModel
from modalities.running_env.cuda_env import CudaEnv
from tests.conftest import _ROOT_DIR

os.environ["LOCAL_RANK"] = "0"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["NNODES"] = "1"
os.environ["NPROC_PER_NODE"] = "1"
os.environ["RDZV_ENDPOINT"] = "0.0.0.0:29502"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"


@pytest.fixture()
def config_file_name() -> str:
    return "config_lorem_ipsum_lora_training.yaml"


@pytest.fixture()
def config_file_path(config_file_name) -> Path:
    config_file_path = _ROOT_DIR / Path("tests/fine_tuning/test_configs/" + config_file_name)
    if not os.path.exists(config_file_path):
        raise FileNotFoundError("Config file doesn't exist")
    return config_file_path


@pytest.fixture
def checkpointing_path(tmp_path):
    return tmp_path / "smol_lora_instruct/"


@pytest.fixture
def main_obj(config_file_path, checkpointing_path):
    main_obj = Main(config_file_path)
    main_obj.config_dict["settings"]["paths"]["checkpointing_path"] = str(checkpointing_path)
    main_obj.config_dict["checkpoint_saving"]["config"]["checkpoint_saving_execution"]["config"]["checkpoint_path"] = (
        str(checkpointing_path)
    )
    return main_obj


@pytest.mark.skipif(
    "RANK" not in os.environ or torch.cuda.device_count() < 1,
    reason="This e2e test requires 1 GPU and a torchrun distributed environment.",
)
def test_lora_model_training_on_one_gpu(main_obj, checkpointing_path):
    with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
        components = main_obj.build_components(components_model_type=TrainingComponentsInstantiationModel)
        main_obj.run(components)

    assert os.path.exists(checkpointing_path)
    checkpoint_files = [
        "model" in path.name or "optimizer" in path.name or path.suffix == ".yaml"
        for path in list(checkpointing_path.glob("*"))[0].glob("*")
    ]
    assert sum(checkpoint_files) == 3, "Output of the test i.e. a model checkpoint was not created!"
