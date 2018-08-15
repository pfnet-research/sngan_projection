import sys
import os

import chainer
import chainer.links as L
from chainer import functions as F
from gen_models.resblocks import Block


class ResNetGeneratorSR(chainer.Chain):
    def __init__(self, ch=64, activation=F.relu, dim_noise=0, distribution="normal", ):
        super(ResNetGeneratorSR, self).__init__()
        initializer = chainer.initializers.GlorotUniform()
        self.activation = activation
        self.distribution = distribution
        self.dim_noise = dim_noise
        with self.init_scope():
            self.block0_1 = Block(3, ch * 4, activation=activation, upsample=False)
            self.block0_2 = Block(ch * 4, ch * 4, activation=activation, upsample=False)
            self.block0_3 = Block(ch * 4, ch * 4, activation=activation, upsample=False)
            self.block0_4 = Block(ch * 4, ch * 4, activation=activation, upsample=False)
            self.block1_1 = Block(ch * 4, ch * 2, activation=activation, upsample=True, dim_noise=self.dim_noise)
            self.block1_2 = Block(ch * 2, ch * 2, activation=activation, upsample=False)
            self.block2_1 = Block(ch * 2, ch, activation=activation, upsample=True, dim_noise=self.dim_noise)
            self.block2_2 = Block(ch, ch, activation=activation, upsample=False)
            self.bout = L.BatchNormalization(ch)
            self.cout = L.Convolution2D(ch, 3, ksize=3, stride=1, pad=1, initialW=initializer)

    def __call__(self, y):
        h = y
        h = self.block0_1(h)
        h = self.block0_2(h)
        h = self.block0_3(h)
        h = self.block0_4(h)
        # Upsample
        h = self.block1_1(h)
        h = self.block1_2(h)
        # Upsample
        h = self.block2_1(h)
        h = self.block2_2(h)
        h = self.bout(h)
        h = self.activation(h)
        h = F.tanh(self.cout(h))
        return h
