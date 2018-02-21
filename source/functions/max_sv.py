import chainer.functions as F
from chainer import cuda


def _l2normalize(v, eps=1e-12):
    norm = cuda.reduce('T x', 'T out',
                       'x * x', 'a + b', 'out = sqrt(a)', 0,
                       'norm_sn')
    div = cuda.elementwise('T x, T norm, T eps',
                           'T out',
                           'out = x / (norm + eps)',
                           'div_sn')
    return div(v, norm(v), eps)


def max_singular_value(W, u=None, Ip=1):
    """
    Apply power iteration for the weight parameter
    """
    if not Ip >= 1:
        raise ValueError("The number of power iterations should be positive integer")

    xp = cuda.get_array_module(W.data)
    if u is None:
        u = xp.random.normal(size=(1, W.shape[0])).astype(xp.float32)
    _u = u
    for _ in range(Ip):
        _v = _l2normalize(xp.dot(_u, W.data), eps=1e-12)
        _u = _l2normalize(xp.dot(_v, W.data.transpose()), eps=1e-12)
    sigma = F.sum(F.linear(_u, F.transpose(W)) * _v)
    return sigma, _u, _v


def max_singular_value_fully_differentiable(W, u=None, Ip=1):
    """
    Apply power iteration for the weight parameter (fully differentiable version)
    """
    if not Ip >= 1:
        raise ValueError("The number of power iterations should be positive integer")

    xp = cuda.get_array_module(W.data)
    if u is None:
        u = xp.random.normal(size=(1, W.shape[0])).astype(xp.float32)
    _u = u
    for _ in range(Ip):
        _v = F.normalize(F.matmul(_u, W), eps=1e-12)
        _u = F.normalize(F.matmul(_v, F.transpose(W)), eps=1e-12)
    _u = F.matmul(_v, F.transpose(W))
    norm = F.sqrt(F.sum(_u ** 2))
    return norm, _l2normalize(_u.data), _v
