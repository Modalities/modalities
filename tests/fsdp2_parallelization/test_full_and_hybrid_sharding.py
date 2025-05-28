import os
import tempfile
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp
import yaml
from pydantic import BaseModel
from torch.distributed.fsdp import FSDPModule as FSDP2

from modalities.__main__ import Main
from modalities.config.config import ProcessGroupBackendType
from modalities.config.pydantic_if_types import PydanticFSDP2ModuleType
from modalities.util import get_local_number_of_trainable_parameters, get_total_number_of_trainable_parameters
from tests.end2end_tests.custom_components import MultiProcessingCudaEnv


@pytest.fixture
def temp_file_path() -> Path:
    # Create a NamedTemporaryFile that persists after closing (delete=False)
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        file_path = tf.name
    try:
        yield Path(file_path)
    finally:
        # Clean up the file after the test
        if os.path.exists(file_path):
            os.remove(file_path)


@pytest.mark.skipif(
    torch.cuda.device_count() < 8,
    reason="This test requires 8 GPUs",
)
class TestFSDP2Sharding:
    def _get_tmp_sharding_config_path(
        self, replication_degree: int, sharding_degree: int, tp_degree: int, temp_file_path: Path
    ) -> Path:
        working_dir = Path(os.path.dirname(__file__))
        config_file_path = working_dir / "../checkpointing/fsdp2_gpt2_config.yaml"

        with open(config_file_path, "r") as file:
            config_string = file.read()
            config_dict = yaml.safe_load(config_string)
            config_dict["device_mesh"]["config"]["data_parallel_replicate_degree"] = replication_degree
            config_dict["device_mesh"]["config"]["data_parallel_shard_degree"] = sharding_degree
            config_dict["device_mesh"]["config"]["tensor_parallel_degree"] = tp_degree

        # save to temporary file
        with open(temp_file_path, "w") as file:
            yaml.dump(config_dict, file)

        return temp_file_path

    def _get_fsdp2_wrapped_model(self, config_file_path: Path) -> FSDP2:
        class ComponentsInstantiationModel(BaseModel):
            initialized_model: PydanticFSDP2ModuleType

        main_obj = Main(config_file_path)
        components: ComponentsInstantiationModel = main_obj.build_components(
            components_model_type=ComponentsInstantiationModel
        )
        return components.initialized_model

    @pytest.mark.parametrize(
        "replication_degree, sharding_degree, tp_degree, world_size",
        [
            (2, 2, 1, 4),
            (2, 2, 2, 8),
            (4, 2, 1, 8),
            (2, 4, 1, 8),
        ],
    )
    def test_sharding(
        self, replication_degree: int, sharding_degree: int, tp_degree: int, world_size: int, temp_file_path: Path
    ):
        tmp_sharding_config_path = self._get_tmp_sharding_config_path(
            replication_degree=replication_degree,
            sharding_degree=sharding_degree,
            tp_degree=tp_degree,
            temp_file_path=temp_file_path,
        )
        mp.spawn(
            self._test_sharding_impl,
            args=(world_size, sharding_degree, tmp_sharding_config_path),
            nprocs=world_size,
            join=True,
        )

    def _test_sharding_impl(
        self,
        process_id: int,
        world_size: int,
        sharding_degree: int,
        gpt2_model_config_path: Path,
    ):
        # wraps the actual test function to be able to run it in a distributed  multiprocessing setup
        with MultiProcessingCudaEnv(
            process_group_backend=ProcessGroupBackendType.nccl,
            global_rank=process_id,
            local_rank=process_id,
            world_size=world_size,
            rdvz_port=22356,
        ):
            fsdp2_wrapped_model = self._get_fsdp2_wrapped_model(gpt2_model_config_path)
            local_num_params = get_local_number_of_trainable_parameters(fsdp2_wrapped_model)
            total_num_params = get_total_number_of_trainable_parameters(fsdp2_wrapped_model)
            assert total_num_params == local_num_params * sharding_degree
