import tempfile
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from modalities.config.config import load_app_config_dict
from modalities.inference.text.inference_component import TextInferenceComponent
from modalities.models.utils import ModelTypeEnum, get_model_from_config
from modalities.tokenization.tokenizer_wrapper import PreTrainedHFTokenizer


@pytest.fixture
def gpt2_model_and_tokenizer():
    config_file_path = Path("tests/test_yaml_configs/gpt2_config_optimizer.yaml")
    config_dict = load_app_config_dict(config_file_path=config_file_path)
    model = get_model_from_config(config=config_dict, model_type=ModelTypeEnum.MODEL)
    tokenizer_path = Path("data/tokenizer/hf_gpt2")
    tokenizer = PreTrainedHFTokenizer(
        pretrained_model_name_or_path=tokenizer_path, max_length=None, truncation=None, padding=False
    )
    return model, tokenizer


@pytest.fixture
def temp_system_prompt_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("You are a helpful AI assistant.")
        temp_path = f.name
    yield temp_path
    Path(temp_path).unlink()


class TestTextInferenceComponent:
    # Test actual inference with real model and tokenizer
    def test_actual_inference_greedy_decoding(self, gpt2_model_and_tokenizer):
        """Test greedy decoding with real model produces deterministic output."""
        model, tokenizer = gpt2_model_and_tokenizer

        component = TextInferenceComponent(
            model=model,
            tokenizer=tokenizer,
            system_prompt_path="",
            chat_template="{user_prompt}",
            prompt_template="{text}",
            sequence_length=20,
            temperature=0.0,  # Greedy decoding
            eod_token="<|endoftext|>",
            device=torch.device("cpu"),
        )

        # Run inference twice with same input
        outputs = []
        for _ in range(2):
            with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                component.generate_tokens("The weather today is")
                outputs.append(mock_stdout.getvalue())
        assert outputs[0] == outputs[1], "Greedy decoding should produce deterministic outputs"

    def test_actual_inference_with_different_temperatures(self, gpt2_model_and_tokenizer):
        """Test that different temperatures produce different outputs."""
        model, tokenizer = gpt2_model_and_tokenizer

        def run_inference_with_temperature(temp):
            component = TextInferenceComponent(
                model=model,
                tokenizer=tokenizer,
                system_prompt_path="",
                chat_template="{user_prompt}",
                prompt_template="{text}",
                sequence_length=15,
                temperature=temp,
                eod_token="<|endoftext|>",
                device=torch.device("cpu"),
            )

            with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                component.generate_tokens("Hello world")
                return mock_stdout.getvalue()

        # Should produce different outputs
        torch.manual_seed(42)
        output_greedy = run_inference_with_temperature(0.0)
        torch.manual_seed(42)
        output_sampling = run_inference_with_temperature(1.0)

        assert output_greedy != output_sampling, "Greedy and sampling outputs should be different"

    def test_run_method_multiple_temperatures(self, gpt2_model_and_tokenizer, temp_system_prompt_file):
        """Test the run() method with multiple temperature inputs."""
        model, tokenizer = gpt2_model_and_tokenizer

        component = TextInferenceComponent(
            model=model,
            tokenizer=tokenizer,
            system_prompt_path=temp_system_prompt_file,
            chat_template="System: {system_prompt}\nUser: {user_prompt}\nAssistant:",
            prompt_template="What is {topic}?",
            sequence_length=20,
            temperature=0.7,
            eod_token="<|endoftext|>",
            device=torch.device("cpu"),
        )
        mock_inputs = ["science", "0.3, 0.8, 1.2", KeyboardInterrupt()]
        with patch("builtins.input", side_effect=mock_inputs):
            with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                component.run()

        output = mock_stdout.getvalue()

        # Verify multiple generations occurred
        assert "(Temperature: 0.3)" in output
        assert "(Temperature: 0.8)" in output
        assert "(Temperature: 1.2)" in output
        assert "🏁 ALL GENERATIONS COMPLETE" in output

    def test_run_method_default_temperature(self, gpt2_model_and_tokenizer, temp_system_prompt_file):
        """Test the run() method with default temperature (empty input)."""
        model, tokenizer = gpt2_model_and_tokenizer

        component = TextInferenceComponent(
            model=model,
            tokenizer=tokenizer,
            system_prompt_path=temp_system_prompt_file,
            chat_template="System: {system_prompt}\nUser: {user_prompt}\nAssistant:",
            prompt_template="What is {topic}?",
            sequence_length=20,
            temperature=0.7,
            eod_token="<|endoftext|>",
            device=torch.device("cpu"),
        )

        mock_inputs = ["machine learning", "", KeyboardInterrupt()]

        with patch("builtins.input", side_effect=mock_inputs):
            with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                component.run()

        output = mock_stdout.getvalue()

        # Verify default temperature was used
        assert "Using default temperature: 0.8" in output
