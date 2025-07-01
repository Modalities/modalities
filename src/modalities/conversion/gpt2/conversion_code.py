import os
import shutil


def _copy_model_files(output_dir: str):
    source_dir = os.path.dirname(__file__)
    modeling_gpt2_path = os.path.join(source_dir, "modeling_gpt2.py")
    configuration_gpt2_path = os.path.join(source_dir, "configuration_gpt2.py")
    shutil.copy(modeling_gpt2_path, output_dir)
    shutil.copy(configuration_gpt2_path, output_dir)


def _change_modalities_import_to_relative_import(output_dir: str):
    target_modeling_file = os.path.join(output_dir, "modeling_gpt2.py")
    with open(target_modeling_file, "r") as file:
        content = file.read()
    content = content.replace("modalities.conversion.gpt2.configuration_gpt2", ".configuration_gpt2")
    with open(target_modeling_file, "w") as file:
        file.write(content)


def transfer_model_code(output_dir: str):
    """Copies the required model code to the output directory and replaces modalities imports.
       This allows the converted model to be used without the modalities package via:
       >>> from transformers import AutoModelForCausalLM
       >>> model = AutoModelForCausalLM.from_pretrained("path/to/converted/model", trust_remote_code=True)

    Args:
        output_dir (str): Directory of the converted model.
    """
    _copy_model_files(output_dir)
    _change_modalities_import_to_relative_import(output_dir)
