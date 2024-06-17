from typing import Dict

import pytest
import torch
import torch.nn as nn

from modalities.checkpointing.torch.torch_checkpoint_loading import TorchCheckpointLoading
from modalities.config.config import PrecisionEnum


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._weights = nn.Linear(2, 3)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        output = self._weights(**inputs)
        return {"output": output}


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="This e2e test requires 1 GPU.")
def test_load_model_checkpoint(tmp_path):
    # After storing the state_dict on disc, the model state does not
    # contain any information about the device or precision
    tmp_file_path = tmp_path / "model_state.pth"

    # model that we checkpoint
    model_1 = DummyModel().to(dtype=PrecisionEnum.BF16.value)

    # models that we load the checkpoint into
    model_2 = DummyModel().to(dtype=PrecisionEnum.FP16.value)
    model_3 = DummyModel().to(dtype=PrecisionEnum.FP16.value)

    # perform checkpointing
    model_state = model_1.state_dict()
    torch.save(model_state, tmp_file_path)

    # load the model checkpoint with different settings
    loaded_model_1: DummyModel = TorchCheckpointLoading(
        device=torch.device("cuda:1"), precision=PrecisionEnum.FP32
    ).load_model_checkpoint(model_2, tmp_file_path)

    assert torch.equal(model_1._weights.weight.to("cuda:1"), loaded_model_1._weights.weight)
    assert torch.equal(model_1._weights.bias.to("cuda:1"), loaded_model_1._weights.bias)

    # since we provided the precision, the model will be loaded with the specified precision
    # even if the state dict contains a different precision.
    assert loaded_model_1._weights.weight.dtype == torch.float32
    assert loaded_model_1._weights.weight.device == torch.device("cuda:1")

    # if we don't specify the precision, the model will be loaded with the precision of the state dict.
    # In this case, BF16 is used as defined for model_1.
    loaded_model_2: DummyModel = TorchCheckpointLoading(device=torch.device("cuda:1")).load_model_checkpoint(
        model_3, tmp_file_path
    )
    assert loaded_model_2._weights.weight.dtype == torch.bfloat16
