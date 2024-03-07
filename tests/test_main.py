import pytest
import torch.cuda

from modalities.__main__ import Main


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="This e2e test requires 1 GPU.")
def test_e2e_training_run_wout_ckpt(monkeypatch, indexed_dummy_data_path, dummy_config):
    # patch in env variables
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", "9948")
    config_dict, config_path = dummy_config
    config_dict["train_dataset"]["config"]["raw_data_path"] = indexed_dummy_data_path.raw_data_path
    main = Main(config_dict, config_path)
    main.run()
