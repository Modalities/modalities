class LazyConcatenation:
    def __init__(self, *lists):
        self.lists = lists
        self._length = sum(map(len, self.lists))

    def __iter__(self):
        for lst in self.lists:
            yield from lst

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        if isinstance(index, int):
            for lst in self.lists:
                if index < len(lst):
                    return lst[index]
                index -= len(lst)
            raise IndexError("list index out of range")
        elif isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return [self[i] for i in range(start, stop, step)]
        else:
            raise TypeError("indices must be integers or slices")
