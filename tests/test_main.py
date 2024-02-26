import pytest
import torch.cuda

from modalities.__main__ import Main


def no_gpu_available() -> bool:
    return not torch.cuda.is_available()


@pytest.mark.skipif(
    no_gpu_available(), reason="This e2e test verifies a GPU-Setup and uses components, which do not support CPU-only."
)
def test_e2e_training_run_wout_ckpt(monkeypatch, indexed_dummy_data_path, dummy_config):
    # patch in env variables
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", "9948")
    config_dict, config_path = dummy_config
    config_dict["train_dataset"]["config"]["raw_data_path"] = indexed_dummy_data_path.raw_data_path
    main = Main(config_dict, config_path)
    main.run()
