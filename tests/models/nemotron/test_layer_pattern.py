import pytest

from modalities.models.nemotron.layer_pattern import (
    LayerSymbol,
    count_layers_by_type,
    get_num_layers,
    parse_layer_pattern,
)

# The published Nemotron-3 Nano 30B-A3B pattern (52 layers).
NEMOTRON_3_NANO_PATTERN = "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"


def test_parse_layer_pattern_returns_one_symbol_per_character():
    assert parse_layer_pattern("ME*-") == [
        LayerSymbol.MAMBA,
        LayerSymbol.MOE,
        LayerSymbol.ATTENTION,
        LayerSymbol.MLP,
    ]


def test_parse_layer_pattern_rejects_empty_pattern():
    with pytest.raises(ValueError, match="must not be empty"):
        parse_layer_pattern("")


@pytest.mark.parametrize("pattern", ["MEX", "M|M", "M/M", "m"])
def test_parse_layer_pattern_rejects_unknown_symbols(pattern):
    with pytest.raises(ValueError, match="Invalid layer symbol"):
        parse_layer_pattern(pattern)


def test_parse_layer_pattern_error_reports_position():
    with pytest.raises(ValueError, match="at position 2"):
        parse_layer_pattern("MEX")


def test_count_layers_by_type_includes_absent_types():
    counts = count_layers_by_type("MME")
    assert counts == {
        LayerSymbol.MAMBA: 2,
        LayerSymbol.MOE: 1,
        LayerSymbol.ATTENTION: 0,
        LayerSymbol.MLP: 0,
    }


def test_nemotron_3_nano_pattern_matches_published_architecture():
    # Model report Table 1 / Figure 2: 52 layers of which 6 are self-attention.
    assert get_num_layers(NEMOTRON_3_NANO_PATTERN) == 52
    counts = count_layers_by_type(NEMOTRON_3_NANO_PATTERN)
    assert counts[LayerSymbol.ATTENTION] == 6
    assert counts[LayerSymbol.MAMBA] == 23
    assert counts[LayerSymbol.MOE] == 23
    assert counts[LayerSymbol.MLP] == 0
