"""Layer pattern parsing for hybrid Mamba-Transformer models.

A hybrid model such as Nemotron-3 Nano is described by a pattern string in which every
character denotes one residual sublayer, e.g. ``"MEM*E"``. Each sublayer is a self-contained
pre-norm residual block (``x = x + f(norm(x))``); there is no combined "attention + MLP"
block as in a classical transformer.

The symbols follow the reference implementation in Megatron-LM
(``megatron/core/models/hybrid/hybrid_layer_allocation.py``) so that pattern strings can be
copied verbatim between the two frameworks.
"""

from enum import Enum


class LayerSymbol(str, Enum):
    """
    Enum of the layer types that a hybrid layer pattern can contain.

    The values match the symbols used by Megatron-LM so that pattern strings are portable.

    Attributes:
        MAMBA (str): A Mamba-2 mixer layer.
        ATTENTION (str): A (grouped-query) self-attention layer.
        MOE (str): A mixture-of-experts feed-forward layer.
        MLP (str): A dense feed-forward layer.
    """

    MAMBA = "M"
    ATTENTION = "*"
    MOE = "E"
    MLP = "-"


def parse_layer_pattern(pattern: str) -> list[LayerSymbol]:
    """
    Parses a hybrid layer pattern string into a list of layer symbols.

    Args:
        pattern (str): The pattern string, e.g. ``"MEM*E"``.

    Raises:
        ValueError: If the pattern is empty or contains an unknown symbol.

    Returns:
        list[LayerSymbol]: One symbol per layer, in model order.
    """
    if len(pattern) == 0:
        raise ValueError("The layer pattern must not be empty.")

    valid_symbols = {symbol.value for symbol in LayerSymbol}
    layer_symbols: list[LayerSymbol] = []
    for position, character in enumerate(pattern):
        if character not in valid_symbols:
            raise ValueError(
                f"Invalid layer symbol '{character}' at position {position} of layer pattern '{pattern}'. "
                f"Valid symbols are {sorted(valid_symbols)}."
            )
        layer_symbols.append(LayerSymbol(character))
    return layer_symbols


def count_layers_by_type(pattern: str) -> dict[LayerSymbol, int]:
    """
    Counts how many layers of each type a pattern contains.

    Args:
        pattern (str): The pattern string, e.g. ``"MEM*E"``.

    Returns:
        dict[LayerSymbol, int]: Counts for every layer type, including types with a count of zero.
    """
    layer_symbols = parse_layer_pattern(pattern)
    counts = {symbol: 0 for symbol in LayerSymbol}
    for layer_symbol in layer_symbols:
        counts[layer_symbol] += 1
    return counts


def get_num_layers(pattern: str) -> int:
    """
    Returns the total number of layers described by a pattern.

    Args:
        pattern (str): The pattern string, e.g. ``"MEM*E"``.

    Returns:
        int: The number of layers.
    """
    return len(parse_layer_pattern(pattern))
