from pathlib import Path

from modalities.__main__ import Main, load_app_config_dict
from modalities.config.config import ProcessGroupBackendType
from modalities.config.instantiation_models import TrainingComponentsInstantiationModel
from modalities.running_env.cuda_env import CudaEnv
from tests.conftest import _ROOT_DIR


def test_e2e_instruction_tuning(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", "9949")

    # Load config
    dummy_config_path = _ROOT_DIR / Path("config_files/training/config_lorem_ipsum_sft.yaml")
    config_dict = load_app_config_dict(dummy_config_path)

    # Disable checkpointing
    config_dict["checkpoint_saving"]["config"]["checkpoint_saving_strategy"]["config"]["k"] = 0
    # Here we need to set it to the batched size of our dataset + 1 to not abort early
    # With the original configuration as above and data prallel of 2 total_steps of 16 per GPU is okay,
    # as the real total_steps (which is 12) is smaller
    config_dict["scheduler"]["config"]["total_steps"] = 24 + 1

    main = Main(dummy_config_path)
    main.config_dict = config_dict

    with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
        components = main.build_components(components_model_type=TrainingComponentsInstantiationModel)
        main.run(components)
