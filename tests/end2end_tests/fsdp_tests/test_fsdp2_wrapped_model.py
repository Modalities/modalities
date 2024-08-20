import os
from pathlib import Path

import debugpy
import pytest
import torch
import torch.cuda
from pydantic import BaseModel

from modalities.__main__ import Main
from modalities.config.config import ProcessGroupBackendType
from modalities.config.pydanctic_if_types import PydanticPytorchModuleType
from modalities.models.gpt2.gpt2_model import GPT2LLM
from modalities.running_env.cuda_env import CudaEnv

rank = int(os.getenv("RANK", 0))

debugpy.listen(5691 + rank)  # You can choose a different port
print("Waiting for debugger to attach...")
debugpy.wait_for_client()

working_dir = Path(os.path.dirname(__file__))


class InstantiationFSDPModel(BaseModel):
    model_wrapped: PydanticPytorchModuleType
    model_raw: PydanticPytorchModuleType


class InstantiationRawModel(BaseModel):
    model_raw: PydanticPytorchModuleType


@pytest.mark.skipif(
    "RANK" not in os.environ or torch.cuda.device_count() < 2,
    reason="This e2e test requires 2 GPUs and a torchrun distributed environment.",
)
def test_fsdp_2_wrapped_model_forward_pass_equivalence():
    config_path = working_dir / "configs/fsdp2_model_config.yaml"
    main = Main(config_path=config_path)

    with CudaEnv(process_group_backend=ProcessGroupBackendType.nccl):
        torch.manual_seed(0)
        components_fsdp: InstantiationFSDPModel = main.build_components(components_model_type=InstantiationFSDPModel)
        torch.manual_seed(0)
        components_raw: InstantiationRawModel = main.build_components(components_model_type=InstantiationRawModel)

        model_raw: GPT2LLM = components_raw.model_raw.to(torch.bfloat16)
        fsdp_2_model = components_fsdp.model_wrapped

        batch_size = 4
        vocab_size = model_raw.transformer.wte.weight.shape[0]
        sequence_length = model_raw.sequence_length
        random_tensor = torch.randint(low=0, high=vocab_size, size=(batch_size, sequence_length))

        input_dict = {model_raw.sample_key: random_tensor}

        output_model_raw = model_raw(input_dict)[model_raw.prediction_key]
        output_model_fsdp_2 = fsdp_2_model(input_dict)[model_raw.prediction_key].cpu()

        # evaluation formula: |actual-expected| <= atol+rtol*|expected|
        torch.testing.assert_close(
            output_model_raw,
            output_model_fsdp_2,
            atol=0.017,  # default for bfloat16: 1e-5
            rtol=0,  # default for bfloat16: 0.016
        )
