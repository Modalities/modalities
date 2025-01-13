from dataclasses import dataclass
from enum import Enum
from typing import Optional


@dataclass
class ReadingJob:
    sample_id: int
    batch_size: int


@dataclass
class ProgressMessage:
    worker_type: Enum
    num_samples: int
    process_type: Optional[str] = None
    process_id: Optional[str] = None
