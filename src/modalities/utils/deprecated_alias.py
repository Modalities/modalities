import copy
import warnings
from typing import Any, Callable

from pydantic import AliasPath, BaseModel, model_validator
from pydantic.aliases import AliasChoices
from pydantic.fields import FieldInfo


def add_deprecated_alias(
    field_name: str, alias: str, warning_message: str | None = None
) -> Callable[[type[BaseModel]], type[BaseModel]]:
    """
    Decorator to add a deprecated alias to a specific field in a Pydantic BaseModel.
    Issues a deprecation warning when the alias is used.

    Args:
        field_name (str): The name of the field to add an alias for
        alias (str): The deprecated alias name to register
        warning_message (str | None): Custom warning message (optional)
    Returns:
        Callable[[type[BaseModel]], type[BaseModel]]: Decorator function
    """

    def decorator(cls: type[BaseModel]) -> type[BaseModel]:
        if not issubclass(cls, BaseModel):
            raise TypeError("Decorator can only be applied to Pydantic BaseModel subclasses")
        if field_name not in cls.model_fields:
            raise ValueError(f"While adding alias to BaseModel: Field '{field_name}' not found in model")

        new_field = _build_new_field_with_alias(cls, field_name, alias)
        cls = _add_field_and_deprecation_validator_to_class(cls, new_field, field_name, alias, warning_message)

        return cls

    return decorator


def _build_new_field_with_alias(cls: type[BaseModel], field_name: str, alias: str) -> FieldInfo:
    field_info = cls.model_fields[field_name]

    aliases = _build_alias_list(alias, field_info)

    new_field = copy.deepcopy(field_info)
    new_field.validation_alias = AliasChoices(*aliases)
    if new_field.alias_priority is None:
        # deprecated alias should have lower priority than original field
        new_field.alias_priority = 2
    return new_field


def _build_alias_list(alias: str, field_info: FieldInfo) -> list[str | AliasPath]:
    aliases: list[str | AliasPath] = [alias]

    # Handle existing aliases
    existing_alias = field_info.validation_alias
    if existing_alias:
        if isinstance(existing_alias, AliasChoices):
            aliases.extend(existing_alias.choices)
        else:
            aliases.append(existing_alias)

    return aliases


def _add_field_and_deprecation_validator_to_class(
    cls: type[BaseModel], new_field: FieldInfo, field_name: str, alias: str, warning_message: str | None
) -> type[BaseModel]:
    cls.model_fields[field_name] = new_field
    cls = _add_deprecation_warning(cls, field_name, alias, warning_message)
    res = cls.model_rebuild(force=True)
    if res is None or not res:
        raise RuntimeError("Failed to rebuild the Pydantic model after adding deprecated alias")
    return cls


def _add_deprecation_warning(
    cls: type[BaseModel], field_name: str, alias: str, warning_message: str | None
) -> type[BaseModel]:
    # Store deprecated aliases info for the validator
    if not hasattr(cls, "_deprecated_aliases"):
        setattr(cls, "_deprecated_aliases", {})
    cls._deprecated_aliases[alias] = {
        "field_name": field_name,
        "warning_message": warning_message or f"Alias '{alias}' is deprecated. Use '{field_name}' instead.",
    }

    @model_validator(mode="before")
    def check_deprecated_aliases(cls: type[BaseModel], data: Any) -> Any:
        if isinstance(data, dict):
            for deprecated_alias, info in getattr(cls, "_deprecated_aliases", {}).items():
                if deprecated_alias in data:
                    warnings.warn(info["warning_message"], DeprecationWarning, stacklevel=3)
        return data

    return type(cls.__name__, (cls,), {"check_deprecated_aliases": check_deprecated_aliases})
