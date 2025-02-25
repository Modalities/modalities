from pathlib import Path

from modalities.conversion.gpt2.conversion_code import transfer_model_code


def test_modeling_gpt2_gets_transferred_with_model_files(tmp_path: Path):
    modeling_gpt2_path = tmp_path / "modeling_gpt2.py"
    assert not modeling_gpt2_path.exists()
    transfer_model_code(tmp_path)
    assert modeling_gpt2_path.exists()


def test_configuration_gpt2_gets_transferred_with_model_files(tmp_path: Path):
    configuration_gpt2_path = tmp_path / "configuration_gpt2.py"
    assert not configuration_gpt2_path.exists()
    transfer_model_code(tmp_path)
    assert configuration_gpt2_path.exists()


def test_transferred_modeling_gpt2_does_not_import_from_modalities(tmp_path: Path):
    transfer_model_code(tmp_path)
    with open(tmp_path / "modeling_gpt2.py") as f:
        text = f.read()
        assert "from modalities" not in text
        assert "import modalities" not in text


def test_transferred_configuration_gpt2_does_not_import_from_modalities(tmp_path: Path):
    transfer_model_code(tmp_path)
    with open(tmp_path / "configuration_gpt2.py") as f:
        text = f.read()
        assert "from modalities" not in text
        assert "import modalities" not in text
