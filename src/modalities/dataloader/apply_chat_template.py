import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple

import jsonlines
from jinja2 import Template
from jinja2.exceptions import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment

from modalities.config.config import load_app_config_dict
from modalities.config.instantiation_models import InstructionTuningInstantiationModel


def apply_chat_template(config_file_path: Path):
    """
    Applies a chat template to the given configuration file.

    Args:
        config_file_path (Path): The path to the configuration file.

    Returns:
        None

    Raises:
        None
    """
    config_dict = load_app_config_dict(config_file_path=config_file_path)
    config = InstructionTuningInstantiationModel(**config_dict)
    instruction_data = _stream_jsonl(config.settings.src_path)
    chat_template = _get_chat_template(config.jinja2_chat_template)

    # we want to have all files of the same hash in the same directory
    dst_path = Path(config.settings.dst_path)
    # similar to github only use the first 7 characters of the hash for readability
    hash_str = _get_hash_sum_sha256_of_file(config_file_path)[:7]
    dst_path = dst_path.parent / f"{config.settings.src_path.stem}_{hash_str}" / dst_path.name
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    _store_config_file_with_hash_suffix(config_file_path, dst_path, hash_str)
    dst_path_with_uuid = dst_path.with_suffix(f".{hash_str}" + "".join(dst_path.suffixes))
    with dst_path_with_uuid.open("w", encoding="utf-8") as output_file:
        for entry in instruction_data:
            conversation = entry[config.settings.conversations_key]
            conversation = _map_conversation_roles(conversation, config.instruction_data_transformation.role_mapping)
            chat = chat_template.render(conversation=conversation, chat_template_data=config.chat_template_data)
            entry["chat"] = chat
            json.dump(entry, output_file, ensure_ascii=False)
            output_file.write("\n")
    print(f"Chat template applied and saved to {dst_path_with_uuid}")


def _get_hash_sum_sha256_of_file(file_path: Path) -> str:
    hash = hashlib.sha256()
    bytes = bytearray(128 * 1024)
    mem_view = memoryview(bytes)
    with file_path.open("rb", buffering=0) as f:
        while n := f.readinto(mem_view):
            hash.update(mem_view[:n])
    return hash.hexdigest()


def _store_config_file_with_hash_suffix(config_file_path: Path, dst_path: Path, uuid_str: str) -> None:
    out_config_file_path = dst_path.parent / f"sft_chat_template_config.{uuid_str}.yaml"
    shutil.copyfile(config_file_path, out_config_file_path)


def _get_chat_template(jinja2_chat_template: str) -> Template:
    # yaml adds a newline character when using the multiline "|" indicator. (with ">" it would add spaces instead)
    # we need to remove those
    chat_template = jinja2_chat_template.replace("}\n{", "}{")
    compiled_chat_template = _compile_jinja_template(chat_template)
    return compiled_chat_template


def _map_conversation_roles(conversation: List[Dict[str, Any]], role_mapping: Dict[str, str]) -> List[Dict[str, Any]]:
    new_conversation = []
    for turn in conversation:
        for key, value in turn.items():
            if key == "role" or key == "from":
                turn[key] = role_mapping[value]
        new_conversation.append(turn)
    return new_conversation


def _stream_jsonl(src_file_path: str) -> Generator[Dict[str, Any], None, None]:
    with jsonlines.open(src_file_path) as reader:
        for obj in reader:
            yield obj


def _compile_jinja_template(chat_template: str) -> Template:
    """Code adapted from
    https://github.com/huggingface/transformers/blob/v4.42.0/src/transformers/tokenization_utils_base.py#L1906
    """

    def raise_exception(message: str):
        raise TemplateError(message)

    def tojson(
        x: Any,
        ensure_ascii: bool = False,
        indent: int | str | None = None,
        separators: Tuple[str, str] | None = None,
        sort_keys: bool = False,
    ):
        # We override the built-in tojson filter because Jinja's default filter escapes HTML characters
        # We also expose some options like custom indents and separators
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    jinja_env.filters["tojson"] = tojson
    jinja_env.globals["raise_exception"] = raise_exception
    return jinja_env.from_string(chat_template)
