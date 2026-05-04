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
from modalities.config.pydantic_if_types import PydanticAppStateType, PydanticDeviceMeshIFType
from modalities.util import get_local_number_of_trainable_parameters, get_total_number_of_trainable_parameters
from modalities.utils.typing_utils import FSDPX
from tests.end2end_tests.custom_components import MultiProcessingCudaEnv
from tests.utility import find_free_port


def test_get_local_number_of_trainable_parameters():
    # Create a simple model with trainable parameters
    model = torch.nn.Sequential(torch.nn.Linear(10, 5), torch.nn.ReLU(), torch.nn.Linear(5, 2))

    # Calculate the expected number of trainable parameters
    expected_params = 10 * 5 + 5 + 5 * 2 + 2  # weights_1 + bias_1 + weights_2 + bias_2 = 67

    # Call the function and check the result
    assert get_local_number_of_trainable_parameters(model) == expected_params


@pytest.fixture
def temporary_folder_path():
    with tempfile.TemporaryDirectory() as tmp_dir_path:
        yield Path(tmp_dir_path)


@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="This test requires 4 GPUs.")
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
        "relative_config_path, sharding_strategy, expected_nr_parameters",
        [
            # FDSP1
            ("test_yaml_configs/config_lorem_ipsum_fsdp1.yaml", "NO_SHARD", 6770176),
            ("test_yaml_configs/config_lorem_ipsum_fsdp1.yaml", "FULL_SHARD", 6770176),
            ("test_yaml_configs/config_lorem_ipsum_fsdp1.yaml", "HYBRID_SHARD", 6770176),
            # FSDP2
            ("test_yaml_configs/config_lorem_ipsum_fsdp2.yaml", "FULL_SHARD", 6770176),
            ("test_yaml_configs/config_lorem_ipsum_fsdp2.yaml", "HYBRID_SHARD", 6770176),
            ("test_yaml_configs/config_lorem_ipsum_fsdp2_pp_tp.yaml", "FULL_SHARD", 6770176 + 6438912),
            # we have extra parameters from the output head since we can't use weight tying with pipeline parallelism
        ],
    )
    def test_get_total_number_of_trainable_parameters_fsdpx(
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
                    config["device_mesh"]["config"]["data_parallel_shard_degree"] = -1
                elif sharding_strategy == "HYBRID_SHARD":
                    assert torch.cuda.device_count() % 2 == 0, (
                        "HYBRID_SHARD test requires even number of GPUs. "
                        f"Current number of GPUs: {torch.cuda.device_count()}"
                    )
                    config["device_mesh"]["config"]["data_parallel_replicate_degree"] = 2
                    config["device_mesh"]["config"]["data_parallel_shard_degree"] = -1
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
            args=(world_size, find_free_port(), tmp_config_file_path, expected_nr_parameters, temporary_folder_path),
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
        temporary_folder_path: Path,
    ):
        class CustomComponentInstantiationModel(BaseModel):
            app_state: PydanticAppStateType
            device_mesh: PydanticDeviceMeshIFType | None = None

        with MultiProcessingCudaEnv(
            process_group_backend=ProcessGroupBackendType.nccl,
            global_rank=process_id,
            local_rank=process_id,
            world_size=world_size,
            rdvz_port=rdvz_port,
        ):
            main_obj = Main(tmp_config_file_path, experiments_root_path=temporary_folder_path)
            components: CustomComponentInstantiationModel = main_obj.build_components(
                components_model_type=CustomComponentInstantiationModel
            )
            wrapped_model = components.app_state.model_parts

            TestUtils._assert_correct_total_number_of_trainable_parameters(
                wrapped_model=wrapped_model,
                expected_nr_parameters=expected_nr_parameters,
                device_mesh=components.device_mesh,
            )

    def _assert_correct_total_number_of_trainable_parameters(
        wrapped_model: FSDPX | list[FSDPX], expected_nr_parameters: int, device_mesh: PydanticDeviceMeshIFType | None
    ):
        nr_parameters = get_total_number_of_trainable_parameters(model=wrapped_model, device_mesh=device_mesh)
        assert nr_parameters == expected_nr_parameters
