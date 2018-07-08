import chainer
from chainer import functions as F
from source.links.sn_embed_id import SNEmbedID
from source.links.sn_linear import SNLinear
from dis_models.resblocks import Block, OptimizedBlock


class SNResNetProjectionDiscriminator(chainer.Chain):
    def __init__(self, ch=64, n_classes=0, activation=F.relu):
        super(SNResNetProjectionDiscriminator, self).__init__()
        self.activation = activation
        initializer = chainer.initializers.GlorotUniform()
        with self.init_scope():
            self.block1 = OptimizedBlock(3, ch)
            self.block2 = Block(ch, ch * 2, activation=activation, downsample=True)
            self.block3 = Block(ch * 2, ch * 4, activation=activation, downsample=True)
            self.block4 = Block(ch * 4, ch * 8, activation=activation, downsample=True)
            self.block5 = Block(ch * 8, ch * 8, activation=activation, downsample=True)
            self.block6 = Block(ch * 8, ch * 16, activation=activation, downsample=True)
            self.block7 = Block(ch * 16, ch * 16, activation=activation, downsample=False)
            self.l8 = SNLinear(ch * 16, 1, initialW=initializer)
            if n_classes > 0:
                self.l_y = SNEmbedID(n_classes, ch * 16, initialW=initializer)

    def __call__(self, x, y=None):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.block6(h)
        h = self.block7(h)
        h = self.activation(h)
        h = F.sum(h, axis=(2, 3))
        output = self.l8(h)
        if y is not None:
            w_y = self.l_y(y)
            output += F.sum(w_y * h, axis=1, keepdims=True)
        return output
