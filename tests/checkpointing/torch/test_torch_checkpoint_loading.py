from typing import Dict

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


def test_load_model_checkpoint(tmp_path):
    tmp_file_path = tmp_path / "model_state.pth"

    model = DummyModel()

    model_state = model.state_dict()
    torch.save(model_state, tmp_file_path)

    loaded_model: DummyModel = TorchCheckpointLoading(
        device=torch.device("cpu"), precision=PrecisionEnum.FP16
    ).load_model_checkpoint(model, tmp_file_path)

    assert torch.equal(model._weights.weight, loaded_model._weights.weight)
    assert torch.equal(model._weights.bias, loaded_model._weights.bias)
