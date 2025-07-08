import numpy as _np

class Tensor:
    def __init__(self, data):
        self.data = _np.asarray(data, dtype=_np.float32)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.data

    def tobytes(self):
        return self.data.tobytes()

    def __add__(self, other):
        if isinstance(other, Tensor):
            other = other.data
        return Tensor(self.data + other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            other = other.data
        return Tensor(self.data - other)

    def __rsub__(self, other):
        if isinstance(other, Tensor):
            other = other.data
        return Tensor(other - self.data)

    def __repr__(self):
        return f"Tensor({self.data!r})"


def randn(*shape):
    return Tensor(_np.random.randn(*shape))

def tensor(data):
    return Tensor(data)

