"""Schema-validates every component in the shipped Nemotron configs.

A config can parse as YAML and still be unrunnable: a component's `config` block may use key names
that its registered pydantic config type does not accept, which the component factory only rejects
when it reaches that node - i.e. minutes into a run, after the model has already been built and
sharded. That is exactly how a wrong `lr_scheduler` key survived review in the 30B config: the
existing test only checked that the file parsed and that a few chosen values were right.

This walks the whole config tree and applies the same key check the component factory applies,
without instantiating anything, so it needs no GPUs and catches the entire class of error.
"""

import os
from pathlib import Path

import pytest

from modalities.config.component_factory import ComponentFactory
from modalities.config.config import load_app_config_dict
from modalities.registry.components import COMPONENTS
from modalities.registry.registry import Registry

REPO_ROOT = Path(__file__).parents[3]
SHIPPED_CONFIGS = [
    REPO_ROOT / "config_files" / "training" / "config_nemotron3_nano_30b_a3b_fsdp2.yaml",
    REPO_ROOT / "config_files" / "training" / "config_lorem_ipsum_nemotron_nano_fsdp2.yaml",
    REPO_ROOT / "config_files" / "training" / "config_fineweb_nemotron_nano_fsdp2.yaml",
    REPO_ROOT / "tests" / "test_yaml_configs" / "nemotron_config_initialization.yaml",
    REPO_ROOT / "tests" / "fsdp2_parallelization" / "nemotron_fsdp2_config.yaml",
]


def _load(config_path: Path) -> dict:
    """Loads a config, supplying the launch environment its resolvers expect."""
    launch_env = {"LOCAL_RANK": "0", "RANK": "0", "WORLD_SIZE": "1", "LOCAL_WORLD_SIZE": "1"}
    previous = {key: os.environ.get(key) for key in launch_env}
    os.environ.update(launch_env)
    try:
        return load_app_config_dict(
            config_file_path=config_path,
            experiment_id="schema_check",
            experiments_root_path=Path("/tmp/modalities_schema_check"),
        )
    finally:
        for key, value in previous.items():
            os.environ.pop(key, None) if value is None else os.environ.__setitem__(key, value)


def _iter_component_nodes(node, path="") -> list[tuple[str, dict]]:
    """Yields every (path, node) in the config tree that declares a component."""
    found = []
    if isinstance(node, dict):
        if "component_key" in node and "variant_key" in node:
            found.append((path or "<root>", node))
        for key, value in node.items():
            found.extend(_iter_component_nodes(value, f"{path}.{key}" if path else str(key)))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.extend(_iter_component_nodes(value, f"{path}[{index}]"))
    return found


@pytest.mark.parametrize("config_path", SHIPPED_CONFIGS, ids=lambda p: p.name)
def test_every_component_config_uses_valid_keys(config_path: Path):
    assert config_path.exists(), config_path
    component_factory = ComponentFactory(registry=Registry(COMPONENTS))
    config_dict = _load(config_path)

    nodes = _iter_component_nodes(config_dict)
    assert nodes, f"no components found in {config_path.name}"

    failures = []
    for path, node in nodes:
        component_key, variant_key = node["component_key"], node["variant_key"]
        try:
            config_type = component_factory.registry.get_config(component_key, variant_key)
        except ValueError as error:
            failures.append(f"{path}: unregistered component {component_key}.{variant_key} ({error})")
            continue

        raw = node.get("config", {})
        if not isinstance(raw, dict):
            failures.append(f"{path}: 'config' is {type(raw).__name__}, expected a mapping")
            continue
        # Nested components and BY_REFERENCE stubs are resolved before validation at runtime, so
        # only the key names of this node are checked here - which is what the factory checks too.
        try:
            component_factory._assert_valid_config_keys(
                component_key=component_key,
                variant_key=variant_key,
                config_dict=raw,
                component_config_type=config_type,
            )
        except ValueError as error:
            first_line = str(error).splitlines()[0]
            failures.append(f"{path} ({component_key}.{variant_key}): {first_line}")

    assert not failures, "invalid component config keys:\n  " + "\n  ".join(failures)


@pytest.mark.parametrize("config_path", SHIPPED_CONFIGS, ids=lambda p: p.name)
def test_required_keys_are_present(config_path: Path):
    # A missing required key is the mirror image of the same bug and equally invisible until runtime.
    component_factory = ComponentFactory(registry=Registry(COMPONENTS))
    config_dict = _load(config_path)

    failures = []
    for path, node in _iter_component_nodes(config_dict):
        component_key, variant_key = node["component_key"], node["variant_key"]
        try:
            config_type = component_factory.registry.get_config(component_key, variant_key)
        except ValueError:
            continue
        raw = node.get("config", {})
        if not isinstance(raw, dict):
            continue
        provided = set(raw)
        for field_name, field in config_type.model_fields.items():
            if not field.is_required():
                continue
            # Reuse the factory's own alias resolution so that deprecated aliases declared via
            # `validation_alias=AliasChoices(...)` (e.g. gradient_clipper's `wrapped_model` for
            # `model_parts`) count as satisfying the field, exactly as they do at runtime.
            accepted_names = component_factory._parse_str_aliases({}, field_name, field)
            if not (accepted_names & provided):
                failures.append(
                    f"{path} ({component_key}.{variant_key}): missing required key "
                    f"'{field_name}' (accepted: {sorted(accepted_names)})"
                )

    assert not failures, "missing required component config keys:\n  " + "\n  ".join(failures)
