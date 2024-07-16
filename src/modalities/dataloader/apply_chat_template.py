import json
from pathlib import Path
from typing import Any, Dict, Generator, List

import jsonlines
from jinja2 import Template
from packaging import version

from modalities.config.config import load_app_config_dict
from modalities.config.sft_config import SFTConfig

# TODO copy and adapt: src.modalities.dataloader.dataset.MemMapDataset
# -> it reads lerge JSONL files, jq-pattern filters and tokenizes
# -> select what to tokenize and what to loss-mask (we dont need to have the b_assistant_token)

# Max idea: select what to tokenize and what to loss-mask (we dont need to have the b_assistant_token) then
# have a collate function which applies the chat template
# after collate the input could be too large; packing is more difficult.
#   --> collate is after batching; packing would introduce dynamic batch size


def apply_chat_template(config_file_path: Path):
    config_dict = load_app_config_dict(config_file_path=config_file_path)
    config = SFTConfig(**config_dict)
    instruction_data = _stream_jsonl(config.settings.src_path)
    chat_template_key = config.settings.chat_template_key
    chat_templates = get_chat_templates(config.jinja2_chat_template_files)

    with open(config.settings.dst_path, "w") as output_file:
        for entry in instruction_data:
            conversation = entry[config.settings.conversations_key]
            conversation = map_roles(conversation, config.instruction_data_transformation.role_mapping)
            if chat_template_key in entry:
                chat_template = chat_templates[chat_template_key]
            else:
                chat_template = chat_templates["default"]

            chat = chat_template.render(conversation=conversation, chat_template_data=config.chat_template_data)
            if not all(special_token in chat for special_token in config.chat_template_data["special_tokens"].values()):
                raise ValueError("Not all special tokens are present in the chat template!")
            entry["chat"] = chat
            json.dump(entry, output_file)
            output_file.write("\n")


def get_chat_templates(jinja2_chat_template_files: Dict[str, str]) -> Dict[str, Template]:
    chat_templates = {}
    for key, file_path in jinja2_chat_template_files.items():
        with Path(file_path).open() as file:
            chat_template = "".join(["".join(line.rsplit("\n", 1)) for line in file.readlines()])
            chat_templates[key] = _compile_jinja_template(chat_template)
    return chat_templates


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
