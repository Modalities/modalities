import os
import tempfile
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp
import yaml
from pydantic import BaseModel

from modalities.__main__ import Main
from modalities.config.config import ProcessGroupBackendType
from modalities.config.pydantic_if_types import PydanticAppStateType
from modalities.util import get_total_number_of_trainable_parameters
from modalities.utils.typing import FSDPX
from tests.end2end_tests.custom_components import MultiProcessingCudaEnv


@pytest.fixture
def temporary_folder_path():
    with tempfile.TemporaryDirectory() as tmp_dir_path:
        yield Path(tmp_dir_path)


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="This test requires 4 GPUs.")
class TestUtils:
    # number of parameters in the model calculated as follows:
    # Embeddings: 128*50304
    # Attention pre-layer norm: 128+128
    # Attention: 3*(128*128+128)
    # Atention projection: 128*128
    # feed forward pre-layer norm: 128+128
    # Feed forward (Swiglu with W, V and W2): 2*(128*256+256) (W, V) and 128*256+128 (W2)
    # lm_head pre-layer norm: 128*128
    # lm_head: 0 since it is tied.
    # Total: 128*50304 + 2 * (128+128 + 3*(128*128+128) + 128*128 + 128+128 + 2*(128*256+256) + 128*256+128)
    # + 128*128 = 6770176

    @staticmethod
    @pytest.mark.parametrize(
        "rdvz_port, relative_config_path, sharding_strategy, expected_nr_parameters",
        [  # FDSP1
            (22370, "../test_yaml_configs/config_lorem_ipsum_fsdp1.yaml", "NO_SHARD", 6770176),
            (22371, "../test_yaml_configs/config_lorem_ipsum_fsdp1.yaml", "FULL_SHARD", 6770176),
            (22372, "../test_yaml_configs/config_lorem_ipsum_fsdp1.yaml", "HYBRID_SHARD", 6770176),
            # FSDP2
            (22374, "../test_yaml_configs/config_lorem_ipsum_fsdp2.yaml", "FULL_SHARD", 6770176),
            (22375, "../test_yaml_configs/config_lorem_ipsum_fsdp2.yaml", "HYBRID_SHARD", 6770176),
        ],
    )
    def test_get_total_number_of_trainable_parameters_fsdpx(
        rdvz_port: int,
        relative_config_path: str,
        temporary_folder_path: Path,
        sharding_strategy: str,
        expected_nr_parameters: int,
    ):
        def load_yaml_config(config_file_path: Path) -> dict:
            with open(config_file_path, "r") as f:
                config = yaml.safe_load(f)
            return config

        def update_config(
            config: dict,
            sharding_strategy: str,
        ) -> dict:
            # sets the correct sharding strategy
            if "device_mesh" in config:  # FSDP2
                if sharding_strategy == "FULL_SHARD":
                    config["device_mesh"]["config"]["data_parallel_replicate_degree"] = 1
                    config["device_mesh"]["config"]["data_parallel_shard_degree"] = torch.cuda.device_count()
                elif sharding_strategy == "HYBRID_SHARD":
                    assert torch.cuda.device_count() % 2 == 0, (
                        "HYBRID_SHARD test requires even number of GPUs. "
                        f"Current number of GPUs: {torch.cuda.device_count()}"
                    )
                    config["device_mesh"]["config"]["data_parallel_replicate_degree"] = 2
                    config["device_mesh"]["config"]["data_parallel_shard_degree"] = torch.cuda.device_count() // 2
                else:
                    raise ValueError(f"Invalid sharding strategy: {sharding_strategy}")
            else:  # FSDP1
                config["wrapped_model"]["config"]["sharding_strategy"] = sharding_strategy
            return config

        def save_yaml_config(config_file_path: Path, config: dict):
            with open(config_file_path, "w") as f:
                yaml.safe_dump(config, f)

        working_dir = Path(os.path.dirname(__file__))
        # load, update and save tmp config
        config_file_path = working_dir / relative_config_path
        config = load_yaml_config(config_file_path=config_file_path)
        config_updated = update_config(config=config, sharding_strategy=sharding_strategy)
        tmp_config_file_path = temporary_folder_path / "config.yaml"
        save_yaml_config(config_file_path=tmp_config_file_path, config=config_updated)

        # run the test in a distributed environment
        world_size = torch.cuda.device_count()
        mp.spawn(
            TestUtils._test_get_total_number_of_trainable_parameters_thread,
            args=(world_size, rdvz_port, tmp_config_file_path, expected_nr_parameters),
            nprocs=world_size,
            join=True,
        )

    @staticmethod
    def _test_get_total_number_of_trainable_parameters_thread(
        process_id: int,
        world_size: int,
        rdvz_port: int,
        tmp_config_file_path: Path,
        expected_nr_parameters: int,
    ):
        class CustomComponentInstantiationModel(BaseModel):
            app_state: PydanticAppStateType

        with MultiProcessingCudaEnv(
            process_group_backend=ProcessGroupBackendType.nccl,
            global_rank=process_id,
            local_rank=process_id,
            world_size=world_size,
            rdvz_port=rdvz_port,
        ):
            main_obj = Main(tmp_config_file_path)
            components: CustomComponentInstantiationModel = main_obj.build_components(
                components_model_type=CustomComponentInstantiationModel
            )
            wrapped_model = components.app_state.model

            TestUtils._assert_correct_total_number_of_trainable_parameters(
                wrapped_model=wrapped_model,
                expected_nr_parameters=expected_nr_parameters,
            )

    def _assert_correct_total_number_of_trainable_parameters(wrapped_model: FSDPX, expected_nr_parameters: int):
        nr_parameters = get_total_number_of_trainable_parameters(wrapped_model)
        assert nr_parameters == expected_nr_parameters
