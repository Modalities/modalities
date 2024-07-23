import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Generator, List

import jsonlines
from jinja2 import Template
from packaging import version

from modalities.config.config import load_app_config_dict
from modalities.config.sft_config import SFTConfig


def apply_chat_template(config_file_path: Path):
    config_dict = load_app_config_dict(config_file_path=config_file_path)
    config = SFTConfig(**config_dict)
    instruction_data = _stream_jsonl(config.settings.src_path)
    chat_template = get_chat_template(config.jinja2_chat_template)

    dst_path = Path(config.settings.dst_path)
    # similar to github only use the first 7 characters of the hash for readability
    hash_str = hash_sum_file_sha256(config_file_path)[:7]
    store_config_file_with_hash(config_file_path, dst_path, hash_str)
    dst_path_with_uuid = dst_path.with_suffix(f".{hash_str}" + "".join(dst_path.suffixes))
    with dst_path_with_uuid.open("w") as output_file:
        for entry in instruction_data:
            conversation = entry[config.settings.conversations_key]
            conversation = map_roles(conversation, config.instruction_data_transformation.role_mapping)
            chat = chat_template.render(conversation=conversation, chat_template_data=config.chat_template_data)
            entry["chat"] = chat
            json.dump(entry, output_file)
            output_file.write("\n")


def hash_sum_file_sha256(file_path: Path) -> str:
    hash = hashlib.sha256()
    bytes = bytearray(128 * 1024)
    mem_view = memoryview(bytes)
    with file_path.open("rb", buffering=0) as f:
        while n := f.readinto(mem_view):
            hash.update(mem_view[:n])
    return hash.hexdigest()


def store_config_file_with_hash(config_file_path: Path, dst_path: Path, uuid_str: str) -> None:
    out_config_file_path = dst_path.parent / f"sft_chat_template_config.{uuid_str}.yaml"
    shutil.copyfile(config_file_path, out_config_file_path)


def get_chat_template(jinja2_chat_template: str) -> Template:
    # yaml adds a newline character when using the multiline "|" indicator. (with ">" it would add spaces instead)
    # we need to remove those
    chat_template = jinja2_chat_template.replace("}\n{", "}{")
    compiled_chat_template = _compile_jinja_template(chat_template)
    return compiled_chat_template


def map_roles(conversation: List[Dict[str, Any]], role_mapping: Dict[str, str]) -> List[Dict[str, Any]]:
    return [{key: role_mapping.get(value, value) for key, value in turn.items()} for turn in conversation]


def _stream_jsonl(src_file_path: str) -> Generator[Dict[str, Any], None, None]:
    with jsonlines.open(src_file_path) as reader:
        for obj in reader:
            yield obj


def _compile_jinja_template(chat_template: str) -> Template:
    """Code taken from
    https://github.com/huggingface/transformers/blob/v4.42.0/src/transformers/tokenization_utils_base.py#L1906
    """
    try:
        import jinja2
        from jinja2.exceptions import TemplateError
        from jinja2.sandbox import ImmutableSandboxedEnvironment
    except ImportError:
        raise ImportError("apply_chat_template requires jinja2 to be installed.")

    if version.parse(jinja2.__version__) < version.parse("3.1.0"):
        raise ImportError(
            "apply_chat_template requires jinja2>=3.1.0 to be installed. Your version is " f"{jinja2.__version__}."
        )

    def raise_exception(message):
        raise TemplateError(message)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        # We override the built-in tojson filter because Jinja's default filter escapes HTML characters
        # We also expose some options like custom indents and separators
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    jinja_env.filters["tojson"] = tojson
    jinja_env.globals["raise_exception"] = raise_exception
    return jinja_env.from_string(chat_template)
