from pathlib import Path

import pytest
import torch

from modalities.__main__ import load_app_config_dict
from modalities.models.vision_transformer.vision_transformer_model import VisionTransformer, VisionTransformerConfig
from tests.conftest import _ROOT_DIR


def test_vision_transformer():
    # Create model
    config_file_path = _ROOT_DIR / Path("tests/models/vision_transformer/vision_transformer_config.yaml")
    config_dict = load_app_config_dict(config_file_path=config_file_path)
    config = VisionTransformerConfig.model_validate(config_dict)
    model = VisionTransformer(**dict(config))

    # Create dummy inputs
    dummy_input_image = torch.randn(1, 3, 224, 224)
    dummy_input = dict(images=dummy_input_image)

    # Create optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Run one training step
    optimizer.zero_grad()
    out = model(dummy_input)
    loss = out["logits"].sum()
    loss.backward()
    optimizer.step()

    # Test outputs
    assert "logits" in out
    assert out["logits"].shape == (1, 1000)


@pytest.mark.parametrize(
    "img_size,patch_size,patch_stride,add_cls_token,target_block_size",
    [
        ((224, 224), 16, 16, True, 197),
        ((224, 224), 16, 16, False, 196),
        ((224, 112), 16, 16, False, 98),
        ((480, 480), 16, 16, False, 900),
        ((480 + 1, 480 + 1), 16, 16, False, 900),
        ((224, 224), 8, 16, True, 197),
        ((224, 224), 16, 8, True, 730),
        ((224, 224), 8, 8, True, 785),
    ],
)
def test_vision_transformer_block_size(img_size, patch_size, patch_stride, add_cls_token, target_block_size):
    block_size = VisionTransformer._calculate_block_size(img_size, patch_size, patch_stride, add_cls_token)
    assert block_size == target_block_size
