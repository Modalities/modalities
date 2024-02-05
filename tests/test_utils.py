from modalities.utils.lazy_concatenation import LazyConcatenation


def test_lazy_concatenation():
    lists = [list(range(5)), list(range(10, 15)), list(range(20, 25))]
    lazy_concat = LazyConcatenation(*lists)
    assert list(lazy_concat) == [*lists[0], *lists[1], *lists[2]]
    assert len(lazy_concat) == 15
    assert lazy_concat[2] == 2
    assert lazy_concat[7] == 12
    assert lazy_concat[12] == 22
    assert lazy_concat[4:7] == [4, 10, 11]
