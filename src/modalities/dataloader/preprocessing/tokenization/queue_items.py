from pathlib import Path
from typing import Optional

from pydantic import BaseModel


class Sample(BaseModel):
    # If the index is not shuffled, then the incrementeal_line_id
    # points to the position in the dataset
    # If the index is shuffled, then the incremental_line_id
    # points to the position in the shuffled index and the
    # shuffled_line_id points to the position in the original index
    incremental_line_id: int
    raw_data_path: Path
    offset: int
    sample_length_in_bytes: int
    content_raw: str | bytes
    content_tokenized: Optional[bytes] = None
    token_size_in_bytes: Optional[int] = None
    shuffled_line_id: Optional[int] = None
