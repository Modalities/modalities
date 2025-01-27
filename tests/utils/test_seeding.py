from pytest import mark

from modalities.utils.seeding import calculate_hashed_seed


@mark.parametrize(
    "input_data, max_seed",
    [
        (["a", "b", "c"], 2**32 - 1),
        (["d", "e", "f"], 2**32 - 1),
        (["g", "hij", "klmnop"], 2**32 - 1),
        (
            [
                "5d3b0e03a13dff183d4d77bc258bec18",
                "5d3b0e03a13dff183d4d77bc258bec18",
                "5d3b0e03a13dff183d4d77bc258bec18",
            ],
            2**32 - 1,
        ),
        (
            [
                "123b0e03a13dff183d4d77bc258bec18",
                "456b0e03a13dff183d4d77bc258bec18",
                "789b0e03a13dff183d4d77bc258bec18",
            ],
            2**32 - 1,
        ),
    ],
)
def test_calculate_seed(input_data: list[str], max_seed: int):
    seed = calculate_hashed_seed(input_data=input_data, max_seed=max_seed)
    print(seed)
    assert seed >= 0
    assert seed < max_seed
