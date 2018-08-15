import os

import numpy as np
from PIL import Image
import chainer
from chainer.dataset import dataset_mixin
import random
from chainer import cuda
import scipy.misc


# The dataset class for super resolution
class ImageNetDatasetSR(chainer.dataset.DatasetMixin):
    def __init__(self, path, root, lr_size=32, hr_size=128, use_label=False, resize_method='bilinear', dequantize=True):
        if not lr_size < hr_size:
            raise ValueError("lr_size should be less than hr_size")
        self.base = chainer.datasets.LabeledImageDataset(path, root)
        self.lr_size = lr_size
        self.hr_size = hr_size
        self.resize_method = resize_method
        self.dequantize = dequantize
        self.use_label = use_label

    def make_image_pairs(self, image):
        c, h, w = image.shape
        if c == 1:
            image = np.concatenate([image, image, image], axis=0)

        image = scipy.misc.imresize(image.transpose(1, 2, 0),
                                    [self.hr_size, self.hr_size], self.resize_method).transpose(2, 0, 1)

        hr_image = image
        lr_image = scipy.misc.imresize(
            hr_image.transpose(1, 2, 0),
            [self.lr_size, self.lr_size],
            self.resize_method).transpose(2, 0, 1)
        # Dequantization
        hr_image = hr_image / 128. - 1.
        lr_image = lr_image / 128. - 1.
        if self.dequantize:
            hr_image += np.random.uniform(size=hr_image.shape, low=0., high=1. / 128)
            lr_image += np.random.uniform(size=lr_image.shape, low=0., high=1. / 128)
        return hr_image, lr_image

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        image, label = self.base[i]
        hr_image, lr_image = self.make_image_pairs(image)
        if self.use_label:
            return hr_image, lr_image, label
        else:
            return hr_image, lr_image


if __name__ == "__main__":
    import glob, os, sys

    root_path = sys.argv[1]
    if len(sys.argv) == 4:
        root_path_val = sys.argv[2]
        anno_path = sys.argv[3]
        preprocess_val = True
    else:
        preprocess_val = False

    dirname_to_label = {}
    with open('dirname_to_label.txt', 'r') as f:
        for line in f:
            dirname, label = line.strip('\n').split(' ')
            dirname_to_label[dirname] = label

    count = 0
    n_image_list = []
    filenames = glob.glob(root_path + '/*/*.JPEG')
    for filename in filenames:
        filename = filename.split('/')
        dirname = filename[-2]
        label = int(dirname_to_label[dirname])
        n_image_list.append([os.path.join(filename[-2], filename[-1]), label])
        count += 1
        if count % 10000 == 0:
            print(count)
    print("Num of examples:{}".format(count))
    n_image_list = np.array(n_image_list, np.str)
    np.savetxt('image_list.txt', n_image_list, fmt="%s")

    if preprocess_val:
        count = 0
        n_image_list = []
        import xml.etree.ElementTree as et

        filenames = glob.glob(root_path_val + '/*.JPEG')
        for filename in filenames:
            filename = filename.split('/')
            image_name = filename[-1].strip('.JPEG')
            e = et.parse(os.path.join(anno_path, image_name + '.xml')).getroot()
            dirname = e.findall('object')[0].findall('name')[0].text
            label = int(dirname_to_label[dirname])
            n_image_list.append([os.path.join(filename[-1]), label])
            count += 1
            if count % 1000 == 0:
                print(count)
        print("Num of examples:{}".format(count))
        n_image_list = np.array(n_image_list, np.str)
        np.savetxt('image_list_val.txt', n_image_list, fmt="%s")
