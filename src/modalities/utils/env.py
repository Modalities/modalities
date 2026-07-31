import os
from typing import Any


class EnvOverride:
    def __init__(self, overrides: dict[str, str]):
        self._overrides = overrides
        self._original: dict[str, str | None] = {}

    def __enter__(self):
        for key, value in self._overrides.items():
            self._original[key] = os.environ.get(key)
            os.environ[key] = value

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any | None):
        for key, value in self._original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
