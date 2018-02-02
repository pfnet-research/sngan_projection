import os, sys
import numpy as np
import argparse
import chainer

base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base, '../'))
from evaluation import load_inception_model

import scipy.ndimage as ndimage
from scipy.misc import imresize

IMAGENET_ROOT_PATH = "/path/to/imagenet/train"
IMAGE_LABEL_LIST_PATH = "/path/to/image_label_list/"
train_filenames_and_labels = np.loadtxt(IMAGE_LABEL_LIST_PATH, dtype=np.str)


def get_imagenet_samples(c):
    RESOLUTION = 128
    images = []
    count = 0
    for filename, label in train_filenames_and_labels:
        filename = filename.split('\'')[1]
        label = label.split('\'')[1]
        if int(label) != c:
            continue
        image = ndimage.imread(os.path.join(IMAGENET_ROOT_PATH, filename))
        image = np.asarray(image, dtype=np.uint8)
        image = imresize(image, (RESOLUTION, RESOLUTION))
        images.append(image)
        count += 1
    # Reference samples
    all_ref_samples = np.stack(images, axis=0).transpose((0, 3, 1, 2)).astype(np.float32)
    return all_ref_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='imagenet')
    parser.add_argument('--stat_dir_path', type=str, default='')
    parser.add_argument('--n_classes', type=int, default=1000)
    parser.add_argument('--tf', action='store_true', default=False)
    args = parser.parse_args()
    chainer.cuda.get_device_from_id(args.gpu).use()
    if args.dataset == 'imagenet':
        get_samples = get_imagenet_samples
    else:
        raise NotImplementedError

    if not os.path.exists(args.stat_dir_path):
        os.makedirs(args.stat_dir_path)
    if args.tf:
        import source.inception.inception_score_tf
        from source.inception.inception_score_tf import get_mean_and_cov as get_mean_cov
    else:
        from evaluation import get_mean_cov
        model = load_inception_model(args.inception_model_path)
    for c in range(args.n_classes):
        print('label:{}'.format(c))
        all_ref_samples = get_samples(c)
        if args.tf:
            mean, cov = get_mean_cov(all_ref_samples)
        else:
            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                mean, cov = get_mean_cov(model, all_ref_samples)
        np.savez(os.path.join(args.stat_dir_path, '{}.npz'.format(int(c))), mean=mean, cov=cov)


if __name__ == '__main__':
    main()
