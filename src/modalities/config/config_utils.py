from typing import Annotated

from hydra._internal.utils import _locate
from pydantic.functional_validators import AfterValidator


def validate_class_path(path: str):
    try:
        _locate(path)
    except Exception as hydra_error:
        raise ValueError(
            f"Could not resolve path to class {path}.",
        ) from hydra_error
    return path


ClassPath = Annotated[str, AfterValidator(validate_class_path)]
