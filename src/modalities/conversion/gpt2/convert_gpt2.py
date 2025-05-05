"""
usage: convert_gpt2.py [-h] [--num_testruns NUM_TESTRUNS] [--device_modalities DEVICE_MODALITIES]
                       [--device_hf DEVICE_HF] modalities_config output_dir

Convert GPT-2 model checkpoint to Huggingface transformers format.

positional arguments:
  modalities_config     Path to the modalities config file.
  output_dir            Directory to save the converted model.

options:
  -h, --help            show this help message and exit
  --num_testruns NUM_TESTRUNS
                        Number of test runs to perform.
  --device_modalities DEVICE_MODALITIES
                        Device for the modalities model.
  --device_hf DEVICE_HF
                        Device for the Hugging Face model.
"""

import argparse
import logging
import os
from pathlib import Path

from modalities.config.config import load_app_config_dict
from modalities.conversion.gpt2.conversion_code import transfer_model_code
from modalities.conversion.gpt2.conversion_model import check_converted_model, convert_model_checkpoint
from modalities.conversion.gpt2.conversion_tokenizer import convert_tokenizer

logger = logging.getLogger(__name__)


def convert_gpt2(
    modalities_config_path: str,
    output_dir: str,
    num_testruns: int = 0,
    device_modalities: str = "cpu",
    device_hf: str = "cpu",
) -> None:
    """Takes a modalities gpt2 model and converts it to a Huggingface transformers model.
       The provided config yaml file should contain the model_raw or model section with the model configuration.
       Additionally, the checkpointed_model section should be present and contain the path to the model checkpoint.
       Optionally, the function can run a number of test runs to compare the converted model with the original one.
       If a tokenizer is specified in the config, it will be converted as well.

    Args:
        modalities_config_path (str): Path to the modalities config file.
        output_dir (str): Directory to save the converted model.
        num_testruns (int, optional): Number of test runs to perform. Defaults to 0.
        device_modalities (str, optional): Device for the modalities model. Defaults to "cpu".
        device_hf (str, optional): Device for the Hugging Face model. Defaults to "cpu".
    """
    modalities_config = load_app_config_dict(Path(modalities_config_path), experiment_id="-1")
    hf_model, modalities_model = convert_model_checkpoint(modalities_config)

    if num_testruns > 0:
        check_converted_model(
            hf_model.to(device_hf),
            modalities_model.to(device_modalities),
            num_testruns,
            modalities_config["model_raw" if "model_raw" in modalities_config else "model"]["config"]["vocab_size"],
        )

    sentence_piece_tokenizer_configs = {
        key: subconfig
        for key, subconfig in modalities_config.items()
        if "component_key" in subconfig
        and subconfig["component_key"] == "tokenizer"
        and subconfig["variant_key"] == "pretrained_sp_tokenizer"
    }

    if len(sentence_piece_tokenizer_configs) > 1:
        raise ValueError(
            "Multiple tokenizer configs found. Please specify only one tokenizer config in the modalities config file."
        )

    if len(sentence_piece_tokenizer_configs) == 1:
        tokenizer_model = modalities_config["tokenizer"]["config"]["tokenizer_model_file"]
        bos_token_id, eos_token_id, pad_token_id, _ = convert_tokenizer(tokenizer_model, output_dir)
        # The values bos=1, eos=2 and pad=None are set by default in the model config (as taken from Llama).
        # Overwrite them, with the actual values from the internal SentencePiece tokenizer.
        # Note, that the LlamaTokenizer wrapping around the SentencePiece tokenizer does not know about these values.
        # The unk token id is not set in the model config.
        hf_model.config.bos_token_id = bos_token_id
        hf_model.config.eos_token_id = eos_token_id
        hf_model.config.pad_token_id = pad_token_id
    else:
        logger.warning("No tokenizer specified in the config. Skipping tokenizer conversion.")
    hf_model.config.auto_map = {
        "AutoConfig": "configuration_gpt2.GPT2Config",
        "AutoModel": "modeling_gpt2.GPT2Model",
        "AutoModelForCausalLM": "modeling_gpt2.GPT2ForCausalLM",
    }
    hf_model.save_pretrained(output_dir)
    transfer_model_code(output_dir)


if __name__ == "__main__":
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"

    parser = argparse.ArgumentParser(description="Convert GPT-2 model checkpoint to Huggingface transformers format.")
    parser.add_argument("modalities_config", type=str, help="Path to the modalities config file.")
    parser.add_argument("output_dir", type=str, help="Directory to save the converted model.")
    parser.add_argument("--num_testruns", type=int, default=0, help="Number of test runs to perform.")
    parser.add_argument("--device_modalities", type=str, default="cpu", help="Device for the modalities model.")
    parser.add_argument("--device_hf", type=str, default="cpu", help="Device for the Hugging Face model.")

    args = parser.parse_args()

    convert_gpt2(
        args.modalities_config,
        args.output_dir,
        args.num_testruns,
        args.device_modalities,
        args.device_hf,
    )
