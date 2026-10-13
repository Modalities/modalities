from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

from modalities.config.config import ConfigDictType, ProcessGroupBackendType, load_app_config_dict
from modalities.conversion.gpt2.conversion_model import (
    _build_single_node_dcp_config,
    check_converted_dcp_model,
    check_converted_model,
)
from modalities.conversion.gpt2.convert_gpt2 import convert_gpt2, convert_gpt2_dcp
from modalities.models.gpt2.gpt2_model import GPT2LLM
from modalities.models.utils import ModelTypeEnum, get_model_from_config
from modalities.running_env.cuda_env import MultiProcessingCudaEnv
from tests.conversion.gpt2.helper import check_same_weight_model

CONVERSION_CASES = [
    pytest.param("gpt2_config_test.yaml", "GPT2ForCausalLM", True, id="layer-norm-gpt2"),
    pytest.param("gpt2_rmsnorm_config_test.yaml", "LlamaForCausalLM", False, id="rms-norm-llama"),
]


DCP_CONVERSION_CASES = [
    pytest.param("gpt2_dcp_config.yaml", "GPT2ForCausalLM", True, id="layer-norm-gpt2-dcp"),
    pytest.param("gpt2_rmsnorm_dcp_config.yaml", "LlamaForCausalLM", False, id="rms-norm-llama-dcp"),
]


@pytest.mark.parametrize(
    ("config_file_name", "expected_model_class_name", "_expects_remote_code"),
    CONVERSION_CASES,
    indirect=["config_file_name"],
)
def test_converting_gpt2_does_not_change_weights(
    converted_model: PreTrainedModel,
    original_model: GPT2LLM,
    expected_model_class_name: str,
    _expects_remote_code: bool,
):
    assert converted_model.__class__.__name__ == expected_model_class_name
    check_same_weight_model(converted_model, original_model)


@pytest.mark.parametrize(
    ("config_file_name", "expected_model_class_name", "_expects_remote_code"),
    CONVERSION_CASES,
    indirect=["config_file_name"],
)
def test_converting_gpt2_does_not_change_outputs(
    converted_model: PreTrainedModel,
    original_model: GPT2LLM,
    vocab_size: int,
    expected_model_class_name: str,
    _expects_remote_code: bool,
):
    assert converted_model.__class__.__name__ == expected_model_class_name
    check_converted_model(
        hf_model=converted_model, modalities_model=original_model, num_testruns=1, vocab_size=vocab_size
    )


@pytest.mark.parametrize(
    ("config_file_name", "expected_model_class_name", "expects_remote_code"),
    CONVERSION_CASES,
    indirect=["config_file_name"],
)
def test_convert_gpt2_saves_expected_model_artifacts(
    run_convert_gpt2: None,
    output_dir: Path,
    expected_model_class_name: str,
    expects_remote_code: bool,
):
    converted_config = AutoConfig.from_pretrained(output_dir, local_files_only=True, trust_remote_code=True)
    converted_model = AutoModelForCausalLM.from_pretrained(output_dir, local_files_only=True, trust_remote_code=True)

    assert converted_model.__class__.__name__ == expected_model_class_name
    assert (output_dir / "modeling_gpt2.py").exists() is expects_remote_code
    assert (output_dir / "configuration_gpt2.py").exists() is expects_remote_code
    assert (getattr(converted_config, "auto_map", None) is not None) is expects_remote_code


@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="This test requires 8 GPUs.")
@pytest.mark.parametrize(
    ("dcp_config_file_name", "expected_model_class_name", "_expects_remote_code"),
    DCP_CONVERSION_CASES,
    indirect=["dcp_config_file_name"],
)
def test_converting_dcp_gpt2_does_not_change_weights(
    converted_dcp_model: PreTrainedModel,
    dcp_checkpoint: str,
    expected_model_class_name: str,
    _expects_remote_code: bool,
):
    new_config: ConfigDictType = _build_single_node_dcp_config(dcp_checkpoint)
    assert converted_dcp_model.__class__.__name__ == expected_model_class_name
    with MultiProcessingCudaEnv(ProcessGroupBackendType.nccl, 0, 0, 1, 24570, device_id=0):
        modalities_model = get_model_from_config(new_config, model_type=ModelTypeEnum.DCP_CHECKPOINTED_MODEL)
        check_same_weight_model(converted_dcp_model, modalities_model)


@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="This test requires 8 GPUs.")
@pytest.mark.parametrize(
    ("dcp_config_file_name", "expected_model_class_name", "_expects_remote_code"),
    DCP_CONVERSION_CASES,
    indirect=["dcp_config_file_name"],
)
def test_converting_dcp_gpt2_does_not_change_outputs(
    run_convert_gpt2_dcp: None,
    output_dir: Path,
    dcp_checkpoint: str,
    expected_model_class_name: str,
    _expects_remote_code: bool,
):
    converted_model = AutoModelForCausalLM.from_pretrained(output_dir, local_files_only=True, trust_remote_code=True)
    assert converted_model.__class__.__name__ == expected_model_class_name
    check_converted_dcp_model(
        hf_model_dir=str(output_dir), dcp_dir=dcp_checkpoint, num_testruns=1, device_id_modalities=0, device_hf="cuda:1"
    )


@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="This test requires 8 GPUs.")
@pytest.mark.parametrize(
    ("dcp_config_file_name", "expected_model_class_name", "expects_remote_code"),
    DCP_CONVERSION_CASES,
    indirect=["dcp_config_file_name"],
)
def test_convert_gpt2_dcp_saves_expected_model_artifacts(
    run_convert_gpt2_dcp: None,
    output_dir: Path,
    expected_model_class_name: str,
    expects_remote_code: bool,
):
    converted_config = AutoConfig.from_pretrained(output_dir, local_files_only=True, trust_remote_code=True)
    converted_model = AutoModelForCausalLM.from_pretrained(output_dir, local_files_only=True, trust_remote_code=True)

    assert converted_model.__class__.__name__ == expected_model_class_name
    assert (output_dir / "modeling_gpt2.py").exists() is expects_remote_code
    assert (output_dir / "configuration_gpt2.py").exists() is expects_remote_code
    assert (getattr(converted_config, "auto_map", None) is not None) is expects_remote_code


def test_convert_gpt2_runs_comparison_and_transfers_code_for_gpt2(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    fake_model = _FakeGPT2Model()
    fake_modalities_model = SimpleNamespace(to=lambda device: f"modalities-on-{device}")
    check_calls = []
    transfer_calls = []

    monkeypatch.setattr("modalities.conversion.gpt2.convert_gpt2.GPT2ForCausalLM", _FakeGPT2Model)
    monkeypatch.setattr(
        "modalities.conversion.gpt2.convert_gpt2.load_app_config_dict",
        lambda *args, **kwargs: {"model": {"config": {"vocab_size": 42}}},
    )
    monkeypatch.setattr(
        "modalities.conversion.gpt2.convert_gpt2.convert_model_checkpoint",
        lambda _config: (fake_model, fake_modalities_model),
    )
    monkeypatch.setattr(
        "modalities.conversion.gpt2.convert_gpt2.check_converted_model",
        lambda *args, **kwargs: check_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        "modalities.conversion.gpt2.convert_gpt2.transfer_model_code",
        lambda output_dir: transfer_calls.append(output_dir),
    )

    convert_gpt2("config.yaml", str(tmp_path), num_testruns=2, device_modalities="cuda:0", device_hf="cpu")

    assert check_calls
    assert fake_model.saved_output_dir == str(tmp_path)
    assert fake_model.config.auto_map["AutoModelForCausalLM"] == "modeling_gpt2.GPT2ForCausalLM"
    assert transfer_calls == [str(tmp_path)]


def test_convert_gpt2_sets_tokenizer_ids_without_transferring_code_for_llama(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    fake_model = _FakeLlamaModel()
    transfer_calls = []

    monkeypatch.setattr("modalities.conversion.gpt2.convert_gpt2.GPT2ForCausalLM", _FakeGPT2Model)
    monkeypatch.setattr(
        "modalities.conversion.gpt2.convert_gpt2.load_app_config_dict",
        lambda *args, **kwargs: {
            "model": {"config": {"vocab_size": 42}},
            "tokenizer": {
                "component_key": "tokenizer",
                "variant_key": "pretrained_sp_tokenizer",
                "config": {"tokenizer_model_file": "tokenizer.model"},
            },
        },
    )
    monkeypatch.setattr(
        "modalities.conversion.gpt2.convert_gpt2.convert_model_checkpoint",
        lambda _config: (fake_model, SimpleNamespace()),
    )
    monkeypatch.setattr(
        "modalities.conversion.gpt2.convert_gpt2.convert_tokenizer",
        lambda *args, **kwargs: (11, 12, 13, None),
    )
    monkeypatch.setattr(
        "modalities.conversion.gpt2.convert_gpt2.transfer_model_code",
        lambda output_dir: transfer_calls.append(output_dir),
    )

    convert_gpt2("config.yaml", str(tmp_path))

    assert fake_model.config.bos_token_id == 11
    assert fake_model.config.eos_token_id == 12
    assert fake_model.config.pad_token_id == 13
    assert not hasattr(fake_model.config, "auto_map")
    assert transfer_calls == []


def test_convert_gpt2_rejects_multiple_tokenizers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr("modalities.conversion.gpt2.convert_gpt2.GPT2ForCausalLM", _FakeGPT2Model)
    monkeypatch.setattr(
        "modalities.conversion.gpt2.convert_gpt2.load_app_config_dict",
        lambda *args, **kwargs: {
            "model": {"config": {"vocab_size": 42}},
            "tokenizer": {
                "component_key": "tokenizer",
                "variant_key": "pretrained_sp_tokenizer",
                "config": {"tokenizer_model_file": "tokenizer.model"},
            },
            "tokenizer_2": {
                "component_key": "tokenizer",
                "variant_key": "pretrained_sp_tokenizer",
                "config": {"tokenizer_model_file": "tokenizer_2.model"},
            },
        },
    )
    monkeypatch.setattr(
        "modalities.conversion.gpt2.convert_gpt2.convert_model_checkpoint",
        lambda _config: (_FakeGPT2Model(), SimpleNamespace()),
    )

    with pytest.raises(ValueError, match="Multiple tokenizer configs found"):
        convert_gpt2("config.yaml", str(tmp_path))


def test_convert_gpt2_dcp_runs_full_flow(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    convert_calls = []
    check_calls = []
    cache_cleared = []
    gc_calls = []

    monkeypatch.setattr(
        "modalities.conversion.gpt2.convert_gpt2.convert_dcp_to_torch",
        lambda distributed_cp_dir, temp_dir, model_key: str(Path(temp_dir) / f"{model_key}.yaml"),
    )
    monkeypatch.setattr(
        "modalities.conversion.gpt2.convert_gpt2.convert_gpt2",
        lambda *args, **kwargs: convert_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        "modalities.conversion.gpt2.convert_gpt2.check_converted_dcp_model",
        lambda *args, **kwargs: check_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        "modalities.conversion.gpt2.convert_gpt2.torch.cuda.empty_cache", lambda: cache_cleared.append(True)
    )
    monkeypatch.setattr("modalities.conversion.gpt2.convert_gpt2.gc.collect", lambda: gc_calls.append(True))

    convert_gpt2_dcp(
        "/tmp/dcp", str(tmp_path), num_testruns=3, device_id_modalities="cuda:2", device_hf="cuda:1", model_key="model"
    )

    assert convert_calls
    assert check_calls
    assert cache_cleared == [True]
    assert gc_calls == [True]


@pytest.fixture
def converted_model(run_convert_gpt2: None, output_dir: Path) -> PreTrainedModel:
    return AutoModelForCausalLM.from_pretrained(output_dir, local_files_only=True, trust_remote_code=True).to(
        dtype=torch.bfloat16
    )


@pytest.fixture
def converted_dcp_model(run_convert_gpt2_dcp: None, output_dir: Path) -> PreTrainedModel:
    return AutoModelForCausalLM.from_pretrained(output_dir, local_files_only=True, trust_remote_code=True)


@pytest.fixture
def run_convert_gpt2(gpt2_config_path: Path, output_dir: Path):
    convert_gpt2(str(gpt2_config_path), str(output_dir))


@pytest.fixture
def run_convert_gpt2_dcp(dcp_checkpoint: str, output_dir: Path):
    convert_gpt2_dcp(dcp_checkpoint, str(output_dir))


@pytest.fixture
def original_model(gpt2_config_path: Path) -> GPT2LLM:
    modalities_config = load_app_config_dict(gpt2_config_path)
    return get_model_from_config(modalities_config, model_type=ModelTypeEnum.CHECKPOINTED_MODEL)


@pytest.fixture
def vocab_size(gpt2_config_path: Path) -> int:
    modalities_config = load_app_config_dict(gpt2_config_path)
    return modalities_config["model_raw" if "model_raw" in modalities_config else "model"]["config"]["vocab_size"]


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    return tmp_path / "output"


class _FakeConfig:
    def __init__(self):
        self.bos_token_id = None
        self.eos_token_id = None
        self.pad_token_id = None


class _FakeBaseModel:
    def __init__(self):
        self.config = _FakeConfig()
        self.saved_output_dir = None
        self.to_calls = []

    def to(self, *args, **kwargs):
        self.to_calls.append((args, kwargs))
        return self

    def save_pretrained(self, output_dir: str):
        self.saved_output_dir = output_dir


class _FakeGPT2Model(_FakeBaseModel):
    pass


class _FakeLlamaModel(_FakeBaseModel):
    pass
