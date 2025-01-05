from abc import ABC
from typing import Any


class ProcessingStrategyIF(ABC):
    def process(self, item: Any) -> dict[str, Any] | None:
        raise NotImplementedError

    def __enter__(self):
        raise NotImplementedError

    def finalize(self):
        raise NotImplementedError

    def __exit__(self, exc_type, exc_value, traceback):
        raise
