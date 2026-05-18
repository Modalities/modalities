from modalities.registry.components import COMPONENTS


def test_gpt2_cp_component_is_registered() -> None:
    assert any(
        component.component_key == "model" and component.variant_key == "gpt2_cp" for component in COMPONENTS
    ), "Expected model variant 'gpt2_cp' to be registered in COMPONENTS."
