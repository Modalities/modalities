from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, FilePath


class Settings(BaseModel):
    src_path: FilePath
    dst_path: Path
    conversations_key: str
    chat_template_key: Optional[str] = None


class InstructionDataTransformation(BaseModel):
    role_mapping: Dict[str, str]


class SFTConfig(BaseModel):
    settings: Settings
    instruction_data_transformation: InstructionDataTransformation
    jinja2_chat_templates: Dict[str, str]
    chat_template_data: Dict[str, Any]
