import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple

from jinja2 import Template
from jinja2.exceptions import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment

from modalities.config.instantiation_models import InstructionTuningDataInstantiationModel, SplitConfig


def split_and_apply_chat_template(config_file_path: Path, config_dict: dict) -> Dict[str, Path]:
    """
    Applies a chat template to the given configuration file.

    Args:
        config_file_path (Path): The path to the configuration file.

    Returns:
        Dict[str, Path]: A dictionary mapping the partition to the output file path.

    Raises:
        Exception: If an error occurs during the application of the chat template.
    """
    config = InstructionTuningDataInstantiationModel(**config_dict)
    instruction_data = _stream_jsonl(config.settings.src_path)
    chat_template = _get_chat_template(config.jinja2_chat_template)

    # we want to have all files of the same hash in the same directory
    dst_path = Path(config.settings.dst_path)
    # similar to github only use the first 7 characters of the hash for readability
    hash_str = _get_hash_sum_sha256_of_file(config_file_path)[:7]
    dst_path = dst_path.parent / f"{config.settings.src_path.stem}_{hash_str}" / dst_path.name
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    _store_config_file_with_hash_suffix(config_file_path, dst_path, hash_str)

    default_suffix = f".{hash_str}" + ".".join(dst_path.suffixes)

    partition_to_out_file_mapping = {}

    partition_to_output_file_path_mapping = {}
    for partition, percentage in config.settings.split_config.splitting.model_dump().items():
        if percentage == 0:
            continue
        out_file_path = dst_path.with_name(f"{dst_path.stem}_{partition}").with_suffix(default_suffix)
        partition_to_output_file_path_mapping[partition] = out_file_path
        partition_to_out_file_mapping[partition] = out_file_path.open("w")

    try:
        partitions_sampled = []
        for entry, partition in _split_streaming_data(data=instruction_data, split_config=config.settings.split_config):
            messages = entry[config.settings.messages_key]
            messages = _map_conversation_roles(messages, config.instruction_data_transformation.role_mapping)
            rendered_messages = chat_template.render(messages=messages, chat_template_data=config.chat_template_data)
            entry["chat"] = rendered_messages
            output_file = partition_to_out_file_mapping[partition]
            partitions_sampled.append(partition)
            json.dump(entry, output_file, ensure_ascii=False)
            output_file.write("\n")
        print(f"Chat template applied and saved to {list(partition_to_output_file_path_mapping.values())}")
        return {
            partition: path
            for partition, path in partition_to_output_file_path_mapping.items()
            if partitions_sampled.count(partition) > 0
        }
    except Exception as e:
        raise e
    finally:
        for file in partition_to_out_file_mapping.values():
            file.close()


def _split_streaming_data(
    data: Generator[Dict[str, Any], None, None], split_config: SplitConfig
) -> Generator[Tuple[Dict[str, Any], str], None, None]:
    random.seed(split_config.seed)
    partitions, weights = list(zip(*split_config.splitting.model_dump().items()))
    for entry in data:
        partition = random.choices(partitions, weights=weights)[0]
        yield (entry, partition)


def _get_hash_sum_sha256_of_file(file_path: Path) -> str:
    hash = hashlib.sha256()
    bytes = bytearray(128 * 1024)
    mem_view = memoryview(bytes)
    with file_path.open("rb", buffering=0) as f:
        while n := f.readinto(mem_view):
            hash.update(mem_view[:n])
    return hash.hexdigest()


def _store_config_file_with_hash_suffix(config_file_path: Path, dst_path: Path, uuid_str: str) -> None:
    out_config_file_path = dst_path.parent / f"instruction_chat_template_config.{uuid_str}.yaml"
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
    with open(src_file_path, "r", encoding="utf-8") as reader:
        for line in reader:
            yield json.loads(line)


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
