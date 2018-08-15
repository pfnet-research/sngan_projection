import chainer
import chainer.links as L
from chainer import functions as F
from gen_models.resblocks import Block
from source.miscs.random_samples import sample_categorical, sample_continuous


class ResNetGenerator(chainer.Chain):
    def __init__(self, ch=64, dim_z=128, bottom_width=4, activation=F.relu, n_classes=0, distribution="normal"):
        super(ResNetGenerator, self).__init__()
        initializer = chainer.initializers.GlorotUniform()
        self.bottom_width = bottom_width
        self.activation = activation
        self.distribution = distribution
        self.dim_z = dim_z
        self.n_classes = n_classes
        with self.init_scope():
            self.l1 = L.Linear(dim_z, (bottom_width ** 2) * ch * 16, initialW=initializer)
            self.block2 = Block(ch * 16, ch * 16, activation=activation, upsample=True, n_classes=n_classes)
            self.block3 = Block(ch * 16, ch * 8, activation=activation, upsample=True, n_classes=n_classes)
            self.block4 = Block(ch * 8, ch * 8, activation=activation, upsample=True, n_classes=n_classes)
            self.block5 = Block(ch * 8, ch * 4, activation=activation, upsample=True, n_classes=n_classes)
            self.block6 = Block(ch * 4, ch * 2, activation=activation, upsample=True, n_classes=n_classes)
            self.block7 = Block(ch * 2, ch, activation=activation, upsample=True, n_classes=n_classes)
            self.b8 = L.BatchNormalization(ch)
            self.l8 = L.Convolution2D(ch, 3, ksize=3, stride=1, pad=1, initialW=initializer)

    def __call__(self, batchsize=64, z=None, y=None, **kwargs):
        if z is None:
            z = sample_continuous(self.dim_z, batchsize, distribution=self.distribution, xp=self.xp)
        if y is None:
            y = sample_categorical(self.n_classes, batchsize, distribution="uniform",
                                   xp=self.xp) if self.n_classes > 0 else None
        if (y is not None) and z.shape[0] != y.shape[0]:
            raise Exception('z.shape[0] != y.shape[0], z.shape[0]={}, y.shape[0]={}'.format(z.shape[0], y.shape[0]))
        h = z
        h = self.l1(h)
        h = F.reshape(h, (h.shape[0], -1, self.bottom_width, self.bottom_width))
        h = self.block2(h, y, **kwargs)
        h = self.block3(h, y, **kwargs)
        h = self.block4(h, y, **kwargs)
        h = self.block5(h, y, **kwargs)
        h = self.block6(h, y, **kwargs)
        h = self.block7(h, y, **kwargs)
        h = self.b8(h)
        h = self.activation(h)
        h = F.tanh(self.l8(h))
        return h
