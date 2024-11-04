from pathlib import Path

from modalities.__main__ import Main, load_app_config_dict
from modalities.config.config import ProcessGroupBackendType
from modalities.config.instantiation_models import TrainingComponentsInstantiationModel
from modalities.running_env.cuda_env import CudaEnv
from tests.conftest import _ROOT_DIR


def test_e2e_instruction_tuning(monkeypatch, tmp_path):
    """
    Run the instruction-tuning training and verify that a model checkpoint was created.
    """
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", "9949")

    # Load config
    dummy_config_path = _ROOT_DIR / Path("tests/config/test_configs/config_sft.yaml")
    config_dict = load_app_config_dict(dummy_config_path)

    # Adapt config for test
    checkpointing_path = tmp_path / "sft_checkpoints/"
    config_dict["settings"]["paths"]["checkpoint_saving_path"] = checkpointing_path.__str__()
    config_dict["checkpoint_saving"]["config"]["checkpoint_saving_execution"]["config"][
        "checkpoint_path"
    ] = checkpointing_path.__str__()
    config_dict["checkpoint_saving"]["config"]["checkpoint_saving_strategy"]["config"]["k"] = 1

    # Here we need to set it to the batched size of our dataset + 1 to not abort early
    # With the original configuration as above and data prallel of 2 total_steps of 16 per GPU is okay,
    # as the real total_steps (which is 12) is smaller
    config_dict["scheduler"]["config"]["total_steps"] = 24 + 1

    main = Main(dummy_config_path)
    main.config_dict = config_dict

    with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
        components = main.build_components(components_model_type=TrainingComponentsInstantiationModel)
        main.run(components)

    checkpoint_files = [
        "model" in path.name or "optimizer" in path.name or path.suffix == ".yaml"
        for path in list(checkpointing_path.glob("*"))[0].glob("*")
    ]
    assert sum(checkpoint_files) == 1, "Output of the test i.e. a model checkpoint was not created!"
