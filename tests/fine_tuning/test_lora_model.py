from pathlib import Path

import pytest
import os

import torch

from modalities.config.config import ProcessGroupBackendType
from modalities.config.instantiation_models import TrainingComponentsInstantiationModel
from modalities.running_env.cuda_env import CudaEnv
from tests.conftest import _ROOT_DIR

from modalities.__main__ import entry_point_run_modalities, Main


@pytest.fixture()
def set_env() -> None:
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ['NNODES'] = '1'
    os.environ['NPROC_PER_NODE'] = '2'
    os.environ['RDZV_ENDPOINT'] = '0.0.0.0:29502'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'



@pytest.fixture()
def config_file_name() -> str:
    return "config_lorem_ipsum_lora_training.yaml"


@pytest.fixture()
def config_file_path(config_file_name: str) -> Path:
    config_file_path = _ROOT_DIR / Path("tests/fine_tuning/test_configs/" + config_file_name)
    return config_file_path


@pytest.fixture
def create_small_dataset():
    import json

    source_file = '/raid/s3/opengptx/alexj/llm_gym/modalities/modalities/tests/fine_tuning/test_data/ultrachat_200k_fastchat.jsonl'
    destination_file = '/raid/s3/opengptx/alexj/llm_gym/modalities/modalities/tests/fine_tuning/test_data/sampled_dataset.jsonl'
    num_lines_to_move = 10000

    with open(source_file, 'r') as src_file:
        lines = [src_file.readline() for _ in range(num_lines_to_move)]

    with open(destination_file, 'w') as dest_file:
        for line in lines:
            dest_file.write(line)

# @pytest.mark.skipif(
#     "RANK" not in os.environ or torch.cuda.device_count() < 2,
#     reason="This e2e test requires 2 GPUs and a torchrun distributed environment.",
# )
def test_lora_model_training(set_env, create_small_dataset, config_file_path: Path):
    assert os.path.exists(config_file_path), "Config file doesn't exist"
    # entry_point_run_modalities(config_file_path=config_file_path)

    main_obj = Main(config_file_path)
    with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
        components = main_obj.build_components(components_model_type=TrainingComponentsInstantiationModel)
        main_obj.run(components)
