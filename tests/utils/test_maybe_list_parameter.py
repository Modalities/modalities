from modalities.utils.maybe_list_parameter import maybe_list_parameter


def test_maybe_list_parameter_works_on_single_item():
    @maybe_list_parameter("x")
    def square(x: int) -> int:
        return x * x

    result = square(x=3)
    assert result == 9


def test_maybe_list_parameter_works_on_list():
    @maybe_list_parameter("x")
    def square(x: int) -> int:
        return x * x

    result = square(x=[1, 2, 3, 4])
    assert result == [1, 4, 9, 16]


def test_maybe_list_parameter_works_on_positional_args():
    @maybe_list_parameter("x")
    def square(x: int) -> int:
        return x * x

    result = square(5)
    assert result == 25

    result = square([2, 3, 4])
    assert result == [4, 9, 16]


def test_maybe_list_parameter_raises_on_missing_parameter():
    try:

        @maybe_list_parameter("y")
        def square(x: int) -> int:
            return x * x

    except ValueError as e:
        assert str(e) == "Parameter 'y' not found in function 'square' signature."
    else:
        assert False, "Expected ValueError was not raised"


def test_maybe_list_parameter_works_on_mixed_args():
    @maybe_list_parameter("x")
    def add_and_square(x: int, y: int) -> int:
        return (x + y) * (x + y)

    result = add_and_square(2, y=3)
    assert result == 25  # (2 + 3)^2

    result = add_and_square([1, 2], y=3)
    assert result == [16, 25]  # (1 + 3)^2, (2 + 3)^2


def test_maybe_list_parameter_works_on_mixed_args_list_positional():
    @maybe_list_parameter("x")
    def add_and_square(x: int, y: int) -> int:
        return (x + y) * (x + y)

    result = add_and_square(4, y=5)
    assert result == 81  # (4 + 5)^2

    result = add_and_square([2, 3], y=5)
    assert result == [49, 64]  # (2 + 5)^2, (3 + 5)^2


def test_maybe_list_parameter_works_on_mixed_args_list_positional_no_kw():
    @maybe_list_parameter("x")
    def add_and_square(x: int, y: int) -> int:
        return (x + y) * (x + y)

    result = add_and_square(6, 7)
    assert result == 169  # (6 + 7)^2

    result = add_and_square([3, 4], 7)
    assert result == [100, 121]  # (3 + 7)^2, (4 + 7)^2


def test_maybe_list_parameter_works_on_empty_list():
    @maybe_list_parameter("x")
    def square(x: int) -> int:
        return x * x

    result = square(x=[])
    assert result == []


def test_maybe_list_parameter_works_on_no_value():
    @maybe_list_parameter("x")
    def square(x: int = 10) -> int:
        return x * x

    result = square()
    assert result == 100


def test_maybe_list_parameter_works_on_no_value_list():
    @maybe_list_parameter("x")
    def square(x: int = 10) -> int:
        return x * x

    result = square(x=[])
    assert result == []


def test_maybe_list_parameter_works_on_no_value_with_other_args():
    @maybe_list_parameter("x")
    def add_and_square(x: int = 10, y: int = 5) -> int:
        return (x + y) * (x + y)

    result = add_and_square(y=3)
    assert result == 169  # (10 + 3)^2


def test_maybe_list_parameter_works_on_no_value_with_other_args_list():
    @maybe_list_parameter("x")
    def add_and_square(x: int = 10, y: int = 5) -> int:
        return (x + y) * (x + y)

    result = add_and_square(x=[], y=3)
    assert result == []


def test_maybe_list_parameter_works_on_multiple_parameters():
    @maybe_list_parameter("x")
    @maybe_list_parameter("y")
    def add(x: int, y: int) -> int:
        return x + y

    result = add(x=[1, 2], y=[10, 20])
    assert result == [[11, 21], [12, 22]]


def test_maybe_list_parameter_works_on_multiple_parameters_mixed():
    @maybe_list_parameter("x")
    @maybe_list_parameter("y")
    def add(x: int, y: int) -> int:
        return x + y

    result = add(x=[1, 2], y=30)
    assert result == [31, 32]

    result = add(x=7, y=[70, 80])
    assert result == [77, 87]


def test_maybe_list_parameter_works_on_multiple_parameters_single():
    @maybe_list_parameter("x")
    @maybe_list_parameter("y")
    def add(x: int, y: int) -> int:
        return x + y

    result = add(x=4, y=6)
    assert result == 10


def test_maybe_list_parameter_calls_apply_to_list_result():
    @maybe_list_parameter("x", apply_to_list_result=sum)
    def square(x: int) -> int:
        return x * x

    result = square(x=[1, 2, 3])
    assert result == 14  # 1^2 + 2^2 + 3^2 = 14


def test_maybe_list_parameter_calls_apply_to_list_input_and_result():
    def apply_func(inputs: list[int], results: list[int]) -> list[int]:
        return [i + r for i, r in zip(inputs, results)]

    @maybe_list_parameter("x", apply_to_list_input_and_result=apply_func)
    def square(x: int) -> int:
        return x * x

    result = square(x=[1, 2, 3])
    assert result == [2, 6, 12]  # [1+1^2, 2+2^2, 3+3^2]
