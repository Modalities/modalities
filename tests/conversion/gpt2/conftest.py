import logging
import multiprocessing as py_mp
import os
import shutil
import traceback
from multiprocessing import Queue
from multiprocessing.managers import ListProxy
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp
from pydantic import BaseModel

from modalities.checkpointing.checkpoint_saving_instruction import CheckpointingInstruction
from modalities.checkpointing.fsdp.fsdp_checkpoint_saving import DCPCheckpointSaving
from modalities.config.component_factory import ComponentFactory
from modalities.config.config import ConfigDictType, ProcessGroupBackendType, load_app_config_dict
from modalities.config.pydantic_if_types import PydanticAppStateType
from modalities.models.gpt2.gpt2_model import GPT2LLM
from modalities.models.utils import ModelTypeEnum, get_model_from_config
from modalities.registry.components import COMPONENTS
from modalities.registry.registry import Registry
from modalities.running_env.cuda_env import MultiProcessingCudaEnv
from modalities.training.training_progress import TrainingProgress
from tests.conftest import _ROOT_DIR
from tests.utility import find_free_port, monitor_child_processes


@pytest.fixture
def gpt2_config_path(
    tmpdir_factory: pytest.TempdirFactory,
    initialized_model: GPT2LLM,
    config_file_path: str,
    corrupt_model_head_key_in_state_dict: bool,
) -> Path:
    tmp_path = tmpdir_factory.mktemp("gpt2_model")
    new_config_filename = tmp_path / "gpt2_config_test.yaml"
    model_path = tmp_path / "model.pth"
    shutil.copy(config_file_path, new_config_filename)
    state_dict = initialized_model.state_dict()
    if corrupt_model_head_key_in_state_dict:
        # Rename the key transformer.lm_head.weight to old_lm_head.weight
        # simulating the old format used in modalities' gpt2 models.
        state_dict["old_lm_head.weight"] = state_dict["transformer.lm_head.weight"]
        del state_dict["transformer.lm_head.weight"]
    torch.save(state_dict, model_path)
    with open(new_config_filename, "r") as file:
        content = file.read()
    content = content.replace("checkpoint_path: null", f"checkpoint_path: {model_path}")
    with open(new_config_filename, "w") as file:
        file.write(content)
    return Path(new_config_filename)


@pytest.fixture(params=[False])
def corrupt_model_head_key_in_state_dict(request: pytest.FixtureRequest) -> bool:
    return request.param


@pytest.fixture()
def initialized_model(set_env: None, modalities_config_dict: ConfigDictType) -> GPT2LLM:
    model = get_model_from_config(config=modalities_config_dict, model_type=ModelTypeEnum.MODEL)
    assert isinstance(model, GPT2LLM)
    return model


@pytest.fixture()
def set_env():
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"


@pytest.fixture()
def modalities_config_dict(config_file_path: Path) -> ConfigDictType:
    return load_app_config_dict(config_file_path=config_file_path)


@pytest.fixture()
def config_file_path(config_file_name: str) -> Path:
    config_file_path = _ROOT_DIR / Path("tests/conversion/test_configs/" + config_file_name)
    return config_file_path


@pytest.fixture()
def config_file_name() -> str:
    return "gpt2_config_test.yaml"


@pytest.fixture()
def dcp_checkpoint(tmpdir_factory: pytest.TempdirFactory, corrupt_model_head_key_in_state_dict: bool) -> str:
    tmp_path = tmpdir_factory.mktemp("dcp_checkpoint_test")
    config_file = _ROOT_DIR / "tests" / "conversion" / "test_configs" / "gpt2_dcp_config.yaml"
    world_size = 8
    port = find_free_port()
    manager = py_mp.Manager()
    try:
        error_queue = manager.Queue()
        return_list = manager.list([None] * world_size)

        proc_ctx = mp.spawn(
            _create_dcp_checkpoint_worker,
            args=(
                world_size,
                port,
                tmp_path,
                corrupt_model_head_key_in_state_dict,
                config_file,
                error_queue,
                return_list,
            ),
            nprocs=world_size,
            join=False,
        )

        monitor_child_processes(manager, error_queue, proc_ctx, shutdown_manager=False)

        checkpoint_path = return_list[0]
        if checkpoint_path is None:
            raise RuntimeError("DCP checkpoint creation failed.")

    finally:
        manager.shutdown()

    yield checkpoint_path


def _create_dcp_checkpoint_worker(
    device_idx: int,
    world_size: int,
    port: int,
    output_dir: str,
    corrupt_model_head_key_in_state_dict: bool,
    config_file: str,
    error_queue: Queue,
    return_list: ListProxy,
):
    with MultiProcessingCudaEnv(
        process_group_backend=ProcessGroupBackendType.nccl,
        global_rank=device_idx,
        local_rank=device_idx,
        world_size=world_size,
        rdvz_port=port,
    ):
        try:
            modalities_config_dict = load_app_config_dict(config_file_path=config_file)
            registry = Registry(COMPONENTS)
            component_factory = ComponentFactory(registry=registry)

            class Components(BaseModel):
                app_state: PydanticAppStateType

            components: Components = component_factory.build_components(
                config_dict=modalities_config_dict, components_model_type=Components
            )
            model: GPT2LLM = components.app_state.model
            if corrupt_model_head_key_in_state_dict and hasattr(model.transformer, "lm_head"):
                # Rename the key transformer.lm_head.weight to old_lm_head.weight
                # simulating the old format used in modalities' gpt2 models.
                model.transformer["old_lm_head"] = model.transformer.lm_head
                del model.transformer["lm_head"]

            experiment_id = "0"
            checkpoint_saving_execution = DCPCheckpointSaving(
                checkpoint_path=Path(output_dir), experiment_id=experiment_id, global_rank=device_idx
            )

            checkpointing_instruction = CheckpointingInstruction(save_current=True, checkpoints_to_delete=[])
            training_progress = TrainingProgress(
                num_seen_steps_current_run=0,
                num_seen_tokens_current_run=0,
                num_target_steps=16,  # dummy value
                num_target_tokens=256,  # dummy value
            )
            checkpoint_saving_execution.run_checkpoint_instruction(
                checkpointing_instruction, training_progress, components.app_state
            )
            # FIXME: Hack to get the checkpoint folder path
            full_path = checkpoint_saving_execution._get_checkpointing_folder_path(
                experiment_id=experiment_id,
                num_seen_steps=training_progress.num_seen_steps_current_run,
                num_seen_tokens=training_progress.num_seen_tokens_current_run,
                num_target_steps=training_progress.num_target_steps,
                num_target_tokens=training_progress.num_target_tokens,
            )
            # Copy yaml config file to output dir
            shutil.copy(config_file, Path(full_path) / "config.yaml")
            return_list[device_idx] = full_path
        except Exception as e:
            tb = traceback.format_exc()
            logging.error(f"Process {device_idx} encountered an error:\n{e}")
            logging.error(tb)
            try:
                error_queue.put((device_idx, tb))
            except Exception:
                logging.error("Failed to put exception info into error queue.")
            os._exit(1)
