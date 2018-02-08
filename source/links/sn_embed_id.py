from chainer.functions.connection import embed_id
from chainer.initializers import normal
from chainer import link
from chainer import variable
from chainer.functions.array.broadcast import broadcast_to
from source.functions.max_sv import max_singular_value
import numpy as np


class SNEmbedID(link.Link):
    """Efficient linear layer for one-hot input.
    This is a link that wraps the :func:`~chainer.functions.embed_id` function.
    This link holds the ID (word) embedding matrix ``W`` as a parameter.
    Args:
        in_size (int): Number of different identifiers (a.k.a. vocabulary
            size).
        out_size (int): Size of embedding vector.
        initialW (2-D array): Initial weight value. If ``None``, then the
            matrix is initialized from the standard normal distribution.
            May also be a callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
        ignore_label (int or None): If ``ignore_label`` is an int value,
            ``i``-th column of return value is filled with ``0``.
        Ip (int): The number of power iteration for calculating the spcetral
            norm of the weights.
        factor (float) : constant factor to adjust spectral norm of W_bar.
    .. seealso:: :func:`chainer.functions.embed_id`
    Attributes:
        W (~chainer.Variable): Embedding parameter matrix.
        W_bar (~chainer.Variable): Spectrally normalized weight parameter.
        u (~numpy.array): Current estimation of the right largest singular vector of W.
        (optional) gamma (~chainer.Variable): the multiplier parameter.
        (optional) factor (float): constant factor to adjust spectral norm of W_bar.
    """

    ignore_label = None

    def __init__(self, in_size, out_size, initialW=None, ignore_label=None, Ip=1, factor=None):
        super(SNEmbedID, self).__init__()
        self.ignore_label = ignore_label
        self.Ip = Ip
        self.factor = factor
        with self.init_scope():
            if initialW is None:
                initialW = normal.Normal(1.0)
            self.W = variable.Parameter(initialW, (in_size, out_size))

        self.u = np.random.normal(size=(1, in_size)).astype(dtype="f")
        self.register_persistent('u')

    @property
    def W_bar(self):
        """
        Spectral Normalized Weight
        """
        sigma, _u, _ = max_singular_value(self.W, self.u, self.Ip)
        if self.factor:
            sigma = sigma / self.factor
        sigma = broadcast_to(sigma.reshape((1, 1)), self.W.shape)
        self.u[:] = _u
        return self.W / sigma

    def __call__(self, x):
        """Extracts the word embedding of given IDs.
        Args:
            x (~chainer.Variable): Batch vectors of IDs.
        Returns:
            ~chainer.Variable: Batch of corresponding embeddings.
        """
        return embed_id.embed_id(x, self.W_bar, ignore_label=self.ignore_label)
