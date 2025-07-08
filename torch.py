import numpy as np

class Tensor(np.ndarray):
    pass

def eye(n):
    return np.eye(n)

def randn(*shape):
    return np.random.randn(*shape)

def allclose(a, b):
    return np.allclose(a, b)

def matmul(a, b):
    return a @ b

__all__ = ['Tensor', 'eye', 'randn', 'allclose', 'matmul']
