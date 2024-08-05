from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel, FilePath


class Settings(BaseModel):
    src_path: FilePath
    dst_path: Path
    conversations_key: str


class InstructionDataTransformation(BaseModel):
    role_mapping: Dict[str, str]


class SFTConfig(BaseModel):
    settings: Settings
    instruction_data_transformation: InstructionDataTransformation
    jinja2_chat_template: str
    chat_template_data: Dict[str, Any]
