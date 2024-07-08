
from pathlib import Path
import pytest

from modalities.models.huggingface_adapters.hf_adapter import HFModelAdapterConfig


@pytest.fixture()
def hf_model_adapter_config() -> HFModelAdapterConfig:
    return HFModelAdapterConfig(config={})


def test_convert_posixpath_to_str(hf_model_adapter_config: HFModelAdapterConfig):
    test_data_to_be_formatted = {
        "key1": Path("test/path/1"),
        "key2": [
            {"key211": Path("test/path/211"), "key212": 1},
            {"key221": 1, "key222": Path("test/path/222")},
        ],
        "key3": 1,
    }
    expected_result = {
        "key1": "test/path/1",
        "key2": [
            {"key211": "test/path/211", "key212": 1},
            {"key221": 1, "key222": "test/path/222"},
        ],
        "key3": 1,
    }
    result = hf_model_adapter_config._convert_posixpath_to_str(test_data_to_be_formatted)
    assert result == expected_result
    