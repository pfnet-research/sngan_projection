import chainer
import numpy as np
from source.links.sn_linear import SNLinear
from source.links.sn_convolution_2d import SNConvolution2D
from chainer import functions as F
from dis_models.resblocks import Block, OptimizedBlock


class SNResNetProjectionDiscriminatorSR(chainer.Chain):
    def __init__(self, ch=64, activation=F.relu):
        super(SNResNetProjectionDiscriminatorSR, self).__init__()
        initializer = chainer.initializers.GlorotUniform()
        self.activation = activation
        with self.init_scope():
            self.block0_1 = OptimizedBlock(3, ch, activation=activation)
            self.block0_2 = Block(ch, ch, activation=activation, downsample=False)
            self.block1_1 = Block(ch, ch * 2, activation=activation, downsample=True)
            self.block1_2 = Block(ch * 2, ch * 2, activation=activation, downsample=False)
            self.c_out_x = SNConvolution2D(ch * 2, 3, 3, stride=1, pad=1, initialW=initializer)
            self.block2 = Block(ch * 2, ch * 4, activation=activation, downsample=True)
            self.block3 = Block(ch * 4, ch * 8, activation=activation, downsample=True)
            self.block4 = Block(ch * 8, ch * 16, activation=activation, downsample=True)
            self.block5 = Block(ch * 16, ch * 16, activation=activation, downsample=False)
            self.lout = SNLinear(ch * 16, 1, initialW=initializer)

    def __call__(self, x, y):
        h = x
        h = self.block0_1(h)
        h = self.block0_2(h)
        h = self.block1_1(h)
        h = self.block1_2(h)
        output = self.c_out_x(h) * y
        # global pool with 1 Lipschitz
        output = F.sum(output, axis=(1, 2, 3)) / np.sqrt(np.prod(np.array(output.shape[1:], np.float32)))
        output = F.expand_dims(output, 1)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.activation(h)
        # global pool with 1 Lipschitz
        h = F.sum(h, axis=(2, 3)) / np.sqrt(np.prod(np.array(h.shape[2:], np.float32)))
        output += self.lout(h)
        return output


class SNResNetConcatDiscriminatorSR(chainer.Chain):
    def __init__(self, ch=64, activation=F.relu):
        super(SNResNetConcatDiscriminatorSR, self).__init__()
        initializer = chainer.initializers.GlorotUniform()
        self.activation = activation
        with self.init_scope():
            self.block0_1 = OptimizedBlock(3, ch, activation=activation)
            self.block0_2 = Block(ch, ch, activation=activation, downsample=False)
            self.block1_1 = Block(ch, ch * 2, activation=activation, downsample=True)
            self.block1_2 = Block(ch * 2, ch * 2, activation=activation, downsample=False)
            self.block2 = Block(ch * 2 + 3, ch * 4, activation=activation, downsample=True)
            self.block3 = Block(ch * 4, ch * 8, activation=activation, downsample=True)
            self.block4 = Block(ch * 8, ch * 16, activation=activation, downsample=True)
            self.block5 = Block(ch * 16, ch * 16, activation=activation, downsample=False)
            self.lout = SNLinear(ch * 16, 1, initialW=initializer)

    def __call__(self, x, y):
        h = x
        h = self.block0_1(h)
        h = self.block0_2(h)
        h = self.block1_1(h)
        h = self.block1_2(h)
        h = F.concat([h, y], axis=1)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.activation(h)
        # global pool with 1 Lipschitz
        h = F.sum(h, axis=(2, 3)) / np.sqrt(np.prod(np.array(h.shape[2:], np.float32)))
        output = self.lout(h)
        return output
