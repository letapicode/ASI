import numpy as _np

class _DummyCuda:
    @staticmethod
    def is_available() -> bool:
        return False

    @staticmethod
    def memory_allocated() -> int:
        return 0

    @staticmethod
    def get_device_properties(_):
        class _P:
            total_memory = 1

        return _P()

    @staticmethod
    def utilization() -> float:
        return 0.0

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

    def __matmul__(self, other):
        if isinstance(other, Tensor):
            other = other.data
        return Tensor(self.data @ other)


def randn(*shape):
    return Tensor(_np.random.randn(*shape))

def eye(n):
    return Tensor(_np.eye(n))

def zeros(*shape, dtype=None):
    return Tensor(_np.zeros(shape, dtype=_np.float32))

def allclose(a, b):
    if isinstance(a, Tensor):
        a = a.data
    if isinstance(b, Tensor):
        b = b.data
    return _np.allclose(a, b)

def matmul(a, b):
    if isinstance(a, Tensor):
        a = a.data
    if isinstance(b, Tensor):
        b = b.data
    return Tensor(a @ b)

float32 = _np.float32
long = int

def tensor(data):
    return Tensor(data)

cuda = _DummyCuda()

__all__ = [
    'Tensor',
    'randn',
    'tensor',
    'eye',
    'zeros',
    'allclose',
    'matmul',
    'float32',
    'long',
    'cuda',
]

