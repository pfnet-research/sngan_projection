import numpy as np
import chainer
from chainer import Variable
from updater import loss_dcgan_dis, loss_dcgan_gen, loss_hinge_dis, loss_hinge_gen


class Updater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.models = kwargs.pop('models')
        self.n_dis = kwargs.pop('n_dis')
        self.loss_type = kwargs.pop('loss_type')
        if self.loss_type == 'dcgan':
            self.loss_gen = loss_dcgan_gen
            self.loss_dis = loss_dcgan_dis
        elif self.loss_type == 'hinge':
            self.loss_gen = loss_hinge_gen
            self.loss_dis = loss_hinge_dis
        else:
            raise NotImplementedError
        super(Updater, self).__init__(*args, **kwargs)

    def get_batch(self, xp):
        batch = self.get_iterator('main').next()
        x = []
        y = []
        for j in range(len(batch)):
            x.append(np.asarray(batch[j][0]).astype("f"))
            y.append(np.asarray(batch[j][1]).astype("f"))
        x_real = Variable(xp.asarray(x))
        y_real = Variable(xp.asarray(y))
        return x_real, y_real

    def update_core(self):
        gen = self.models['gen']
        dis = self.models['dis']
        gen_optimizer = self.get_optimizer('opt_gen')
        dis_optimizer = self.get_optimizer('opt_dis')
        xp = gen.xp
        for i in range(self.n_dis):
            if i == 0:
                _, y_real = self.get_batch(xp)
                x_fake = gen(y_real)
                dis_fake = dis(x_fake, y=y_real)
                loss_gen = self.loss_gen(dis_fake=dis_fake)
                gen.cleargrads()
                loss_gen.backward()
                gen_optimizer.update()
                chainer.reporter.report({'loss_gen': loss_gen})

            x_real, y_real = self.get_batch(xp)
            dis_real = dis(x_real, y=y_real)
            x_fake = gen(y_real)
            dis_fake = dis(x_fake, y=y_real)
            x_fake.unchain_backward()

            loss_dis = self.loss_dis(dis_fake=dis_fake, dis_real=dis_real)
            dis.cleargrads()
            loss_dis.backward()
            dis_optimizer.update()
            chainer.reporter.report({'loss_dis': loss_dis})
