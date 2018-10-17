import chainer
from chainer.dataset import dataset_mixin

class Cifar10Dataset(dataset_mixin.DatasetMixin):
    def __init__(self, test=False):
        d_train, d_test = chainer.datasets.get_cifar10(ndim=3, withlabel=True, scale=1.0)
        if test:
            self.dset = d_test
        else:
            self.dset = d_train

        print("load cifar-10.  shape: ", len(self))

    def __len__(self):
        return len(self.dset)

    def get_example(self, i):
        return self.dset[i][0] * 2. - 1., self.dset[i][1]

