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
    dummy_config_path = _ROOT_DIR / Path("tests/config/test_configs/config_lorem_ipsum_instruct_fsdp1.yaml")
    config_dict = load_app_config_dict(dummy_config_path, experiment_id="test_e2e_instruction_tuning")

    # Adapt config for test
    checkpointing_path = tmp_path / "sft_checkpoints/"
    config_dict["settings"]["paths"]["checkpoint_saving_path"] = checkpointing_path.__str__()
    config_dict["checkpoint_saving"]["config"]["checkpoint_saving_execution"]["config"][
        "checkpoint_path"
    ] = checkpointing_path.__str__()

    with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
        main = Main(dummy_config_path)
        main.config_dict = config_dict
        components = main.build_components(components_model_type=TrainingComponentsInstantiationModel)
        main.run(components)

    checkpoint_files = [
        ("model" in path.name or "optimizer" in path.name) and path.suffix == ".bin"
        for path in list(checkpointing_path.glob("*"))[0].glob("*")
    ]
    assert sum(checkpoint_files) == 2, "Output of the test i.e. a model checkpoint and optimizer state was not created!"
