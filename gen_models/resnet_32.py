import chainer
import chainer.links as L
from chainer import functions as F
from gen_models.resblocks import Block
from source.miscs.random_samples import sample_categorical, sample_continuous


class ResNetGenerator(chainer.Chain):
    def __init__(self, ch=256, dim_z=128, bottom_width=4, activation=F.relu, n_classes=0, distribution="normal"):
        super(ResNetGenerator, self).__init__()
        self.bottom_width = bottom_width
        self.activation = activation
        self.distribution = distribution
        self.dim_z = dim_z
        self.n_classes = n_classes
        with self.init_scope():
            self.l1 = L.Linear(dim_z, (bottom_width ** 2) * ch, initialW=chainer.initializers.GlorotUniform())
            self.block2 = Block(ch, ch, activation=activation, upsample=True, n_classes=n_classes)
            self.block3 = Block(ch, ch, activation=activation, upsample=True, n_classes=n_classes)
            self.block4 = Block(ch, ch, activation=activation, upsample=True, n_classes=n_classes)
            self.b5 = L.BatchNormalization(ch)
            self.c5 = L.Convolution2D(ch, 3, ksize=3, stride=1, pad=1, initialW=chainer.initializers.GlorotUniform())

    def sample_z(self, batchsize=64):
        return sample_continuous(self.dim_z, batchsize, distribution=self.distribution, xp=self.xp)

    def sample_y(self, batchsize=64):
        return sample_categorical(self.n_classes, batchsize, distribution="uniform", xp=self.xp)

    def __call__(self, batchsize=64, z=None, y=None):
        if z is None:
            z = sample_continuous(self.dim_z, batchsize, distribution=self.distribution, xp=self.xp)
        if y is None:
            y = sample_categorical(self.n_classes, batchsize, distribution="uniform",
                                   xp=self.xp) if self.n_classes > 0 else None
        if (y is not None) and z.shape[0] != y.shape[0]:
            raise ValueError('z.shape[0] != y.shape[0]')
        h = z
        h = self.l1(h)
        h = F.reshape(h, (h.shape[0], -1, self.bottom_width, self.bottom_width))
        h = self.block2(h, y)
        h = self.block3(h, y)
        h = self.block4(h, y)
        h = self.b5(h)
        h = self.activation(h)
        h = F.tanh(self.c5(h))
        return h
