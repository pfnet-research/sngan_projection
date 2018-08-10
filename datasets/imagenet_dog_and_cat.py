import sys
import numpy as np
import chainer
import random
import scipy.misc

classes_dog_and_cat = np.arange(151, 294, dtype=np.int32)
class_to_index = dict()
for i, c in enumerate(classes_dog_and_cat):
    class_to_index[c] = i


class ImageNetDogAndCatDataset(chainer.dataset.DatasetMixin):
    def __init__(self, path, root, size=128, resize_method='bilinear', augmentation=False, crop_ratio=0.9):
        self.base = chainer.datasets.LabeledImageDataset(path, root)
        self.size = size
        self.resize_method = resize_method
        self.augmentation = augmentation
        self.crop_ratio = crop_ratio

    def __len__(self):
        return len(self.base)

    def transform(self, image):
        c, h, w = image.shape
        if c == 1:
            image = np.concatenate([image, image, image], axis=0)
        short_side = h if h < w else w
        if self.augmentation:
            crop_size = int(short_side * self.crop_ratio)
            # Randomly crop a region and flip the image
            top = random.randint(0, h - crop_size - 1)
            left = random.randint(0, w - crop_size - 1)
            if random.randint(0, 1):
                image = image[:, :, ::-1]
        else:
            crop_size = short_side
            # Crop the center
            top = (h - crop_size) // 2
            left = (w - crop_size) // 2
        bottom = top + crop_size
        right = left + crop_size

        image = image[:, top:bottom, left:right]
        _, h, w = image.shape
        image = scipy.misc.imresize(image.transpose(1, 2, 0),
                                    [self.size, self.size], self.resize_method).transpose(2, 0, 1)
        image = image / 128. - 1.
        image += np.random.uniform(size=image.shape, low=0., high=1. / 128)
        return image

    def get_example(self, i):
        image, label = self.base[i]
        image = self.transform(image)
        index = class_to_index[int(label)]
        return image, index


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
        if label in classes_dog_and_cat:
            n_image_list.append([os.path.join(filename[-2], filename[-1]), label])
            count += 1
            if count % 10000 == 0:
                print(count)
    print("Num of examples:{}".format(count))
    n_image_list = np.array(n_image_list, np.str)
    np.savetxt('image_list_dog_and_cat.txt', n_image_list, fmt="%s")

    if preprocess_val:
        count = 0
        n_image_list = []
        import xml.etree.ElementTree as et
        filenames = glob.glob(root_path_val + '/*.JPEG')
        for filename in filenames:
            filename = filename.split('/')
            image_name = filename[-1].strip('.JPEG')
            e = et.parse(os.path.join(anno_path, image_name+'.xml')).getroot()
            dirname = e.findall('object')[0].findall('name')[0].text
            label = int(dirname_to_label[dirname])
            if label in classes_dog_and_cat:
                n_image_list.append([os.path.join(filename[-1]), label])
                count += 1
                if count % 1000 == 0:
                    print(count)
        print("Num of examples:{}".format(count))
        n_image_list = np.array(n_image_list, np.str)
        np.savetxt('image_list_dog_and_cat_val.txt', n_image_list, fmt="%s")

