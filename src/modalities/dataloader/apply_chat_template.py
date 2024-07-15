import json
from pathlib import Path

import jsonlines
from packaging import version

from modalities.config.config import load_app_config_dict

# TODO copy and adapt: src.modalities.dataloader.dataset.MemMapDataset
# -> it reads lerge JSONL files, jq-pattern filters and tokenizes
# -> select what to tokenize and what to loss-mask (we dont need to have the b_assistant_token)

# Max idea: select what to tokenize and what to loss-mask (we dont need to have the b_assistant_token) then
# have a collate function which applies the chat template
# after collate the input could be too large; packing is more difficult.
#   --> collate is after batching; packing would introduce dynamic batch size


def apply_chat_template(config_file_path: Path):
    config_dict = load_app_config_dict(config_file_path=config_file_path)
    instruction_data = _stream_jsonl(config_dict["settings"]["src_path"])
    chat_template = _compile_jinja_template(config_dict["chat_template"].replace("}\n{", "}{"))
    conversations_key = config_dict["settings"]["conversations_key"]
    role_mapping = config_dict["instruction_data_transformation"]["role_mapping"]
    output_file_path = config_dict["settings"]["dst_path"]
    with open(output_file_path, "w") as output_file:
        for entry in instruction_data:
            conversation = entry[conversations_key]
            conversation = map_roles(conversation, role_mapping)
            chat = chat_template.render(conversation=conversation, chat_template_data=config_dict["chat_template_data"])
            if not all(
                special_token in chat for special_token in config_dict["chat_template_data"]["special_tokens"].values()
            ):
                raise ValueError("Not all special tokens are present in the chat template!")
            entry["chat"] = chat
            json.dump(entry, output_file)
            output_file.write("\n")


def map_roles(conversation, role_mapping):
    return [{key: role_mapping.get(value, value) for key, value in turn.items()} for turn in conversation]


def _stream_jsonl(src_file_path):
    with jsonlines.open(src_file_path) as reader:
        for obj in reader:
            yield obj


def _compile_jinja_template(chat_template):
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
