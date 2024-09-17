from pathlib import Path

import pytest
import torch

from modalities.__main__ import load_app_config_dict
from modalities.models.vision_transformer.vision_transformer_model import VisionTransformer, VisionTransformerConfig
from tests.conftest import _ROOT_DIR


@pytest.mark.parametrize(
    "input,sample_key,n_classes,num_video_frames,add_cls_token,out_put",
    [
        (torch.randn(1, 3, 224, 224), "images", 1000, 1, True, (1, 1000)),
        (torch.randn(1, 3, 224, 224), "images", None, 1, True, (1, 197, 768)),
        (torch.randn(1, 3, 224, 224), "images", None, 1, False, (1, 196, 768)),
        (torch.randn(1, 3, 224, 224), "images", 1000, 1, False, (1, 1000)),
        (torch.randn(1, 16, 3, 224, 224), "videos", 1000, 16, True, (1, 1000)),
        (torch.randn(1, 16, 3, 224, 224), "videos", None, 16, True, (1, 65, 768)),
        (torch.randn(1, 16, 3, 224, 224), "videos", None, 16, False, (1, 64, 768)),
        (torch.randn(1, 16, 3, 224, 224), "videos", 1000, 16, False, (1, 1000)),
    ],
)
def test_vision_transformer(input, sample_key, n_classes, num_video_frames, add_cls_token, out_put):
    # Create model
    config_file_path = _ROOT_DIR / Path("tests/models/vision_transformer/vision_transformer_config.yaml")
    config_dict = load_app_config_dict(config_file_path=config_file_path)
    config = VisionTransformerConfig.model_validate(config_dict)
    config.sample_key = sample_key
    config.n_classes = n_classes
    config.num_video_frames = num_video_frames
    config.add_cls_token = add_cls_token

    model = VisionTransformer(**dict(config))

    # Create dummy inputs
    dummy_input = {sample_key: input}

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
    assert out["logits"].shape == out_put


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
