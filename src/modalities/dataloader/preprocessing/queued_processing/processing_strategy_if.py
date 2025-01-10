from abc import ABC
from typing import Any, Optional


class ProcessingStrategyIF(ABC):
    def process(self, item: Optional[Any] = None) -> dict[str, Any]:
        raise NotImplementedError

    def __enter__(self):
        raise NotImplementedError

    def finalize(self):
        raise NotImplementedError

    def __exit__(self, exc_type, exc_value, traceback):
        raise NotImplementedError
