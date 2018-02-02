import numpy as np
from chainer.functions.connection import convolution_nd
from chainer import initializers
from chainer import link
from chainer.utils import conv_nd
from chainer import variable
from chainer.functions.array.broadcast import broadcast_to

from source.functions.max_sv import max_singular_value


class SNConvolutionND(link.Link):
    """N-dimensional convolution layer.
    This link wraps the :func:`~chainer.functions.convolution_nd` function and
    holds the filter weight and bias vector as parameters.
    Args:
        ndim (int): Number of spatial dimensions.
        in_channels (int): Number of channels of input arrays.
        out_channels (int): Number of channels of output arrays.
        ksize (int or tuple of ints): Size of filters (a.k.a. kernels).
            ``ksize=k`` and ``ksize=(k, k, ..., k)`` are equivalent.
        stride (int or tuple of ints): Stride of filter application.
            ``stride=s`` and ``stride=(s, s, ..., s)`` are equivalent.
        pad (int or tuple of ints): Spatial padding width for input arrays.
            ``pad=p`` and ``pad=(p, p, ..., p)`` are equivalent.
        nobias (bool): If ``True``, then this function does not use the bias.
        initialW (array): Initial weight array. If ``None``, the default
            initializer is used.
            May be a callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
        initial_bias (array): Initial bias vector. If ``None``, the bias is
            set to zero.
            May be a callable that takes ``numpy.ndarray`` or ``cupy.ndarray``
            and edits its value.
        cover_all (bool): If ``True``, all spatial locations are convoluted
            into some output pixels. It may make the output size larger.
            ``cover_all`` needs to be ``False`` if you want to use cuDNN.
        use_gamma (bool): If true, apply scalar multiplication to the
            normalized weight (i.e. reparameterize).
        Ip (int): The number of power iteration for calculating the spcetral
            norm of the weights.
        factor (float) : constant factor to adjust spectral norm of W_bar.
    .. seealso::
        See :func:`~chainer.functions.convolution_nd` for the definition of
        N-dimensional convolution. See
        :func:`~chainer.functions.convolution_2d` for the definition of
        two-dimensional convolution.
    Attributes:
        W (~chainer.Variable): Weight parameter.
        W_bar (~chainer.Variable): Spectrally normalized weight parameter.
        b (~chainer.Variable): Bias parameter. If ``initial_bias`` is ``None``,
            set to ``None``.
        u (~numpy.array): Current estimation of the right largest singular vector of W.
        (optional) gamma (~chainer.Variable): the multiplier parameter.
        (optional) factor (float): constant factor to adjust spectral norm of W_bar.
    """

    def __init__(self, ndim, in_channels, out_channels, ksize, stride=1, pad=0,
                 nobias=False, initialW=None, initial_bias=None,
                 cover_all=False, use_gamma=False, Ip=1, factor=None):
        super(SNConvolutionND, self).__init__()
        ksize = conv_nd.as_tuple(ksize, ndim)
        self.stride = stride
        self.pad = pad
        self.cover_all = cover_all
        self.use_gamma = use_gamma
        self.Ip = Ip
        self.u = np.random.normal(size=(1, out_channels)).astype(dtype="f")
        self.register_persistent('u')
        self.factor = factor
        with self.init_scope():
            W_shape = (out_channels, in_channels) + ksize
            self.W = variable.Parameter(
                initializers._get_initializer(initialW), W_shape)

            if nobias:
                self.b = None
            else:
                if initial_bias is None:
                    initial_bias = 0
                initial_bias = initializers._get_initializer(initial_bias)
                self.b = variable.Parameter(initial_bias, out_channels)

            if self.use_gamma:
                W_mat = self.W.data.reshape(self.W.shape[0], -1)
                _, s, _ = np.linalg.svd(W_mat)
                self.gamma = variable.Parameter(s[0], (1,) * len(self.W.shape))

    @property
    def W_bar(self):
        """
        Spectral Normalized Weight
        """
        W_mat = self.W.reshape(self.W.shape[0], -1)
        sigma, _u, _ = max_singular_value(W_mat, self.u, self.Ip)
        if self.factor:
            sigma = sigma / self.factor
        sigma = broadcast_to(sigma.reshape([1] * len(self.W.shape)), self.W.shape)
        self.u[:] = _u

        if hasattr(self, 'gamma'):
            return broadcast_to(self.gamma, self.W.shape) * self.W / sigma
        else:
            return self.W / sigma

    def __call__(self, x):
        """Applies N-dimensional convolution layer.
        Args:
            x (~chainer.Variable): Input image.
        Returns:
            ~chainer.Variable: Output of convolution.
        """
        return convolution_nd.convolution_nd(
            x, self.W_bar, self.b, self.stride, self.pad, cover_all=self.cover_all)
