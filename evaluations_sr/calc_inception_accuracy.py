import os, sys, time
import shutil
import numpy as np
import argparse
import chainer
from chainer import functions as F
from chainer import cuda

import yaml

base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base, '../'))

import source.yaml_utils as yaml_utils
from scipy.misc import imresize
from extentions_for_eval import load_inception_model


def load_models(config):
    gen_conf = config.models['generator']
    gen = yaml_utils.load_model(gen_conf['fn'], gen_conf['name'], gen_conf['args'])
    return gen


def load_dataset_eval(config):
    dataset = yaml_utils.load_module(config.dataset_eval['dataset_fn'],
                                     config.dataset_eval['dataset_name'])
    return dataset(**config.dataset_eval['args'])


def get_batch_sr_with_label(iterator, xp):
    batch = iterator.next()
    batchsize = len(batch)
    hr_image = []
    lr_image = []
    label = []
    if not len(batch[0]) == 3:
        raise ValueError('missing label data in a batch')
    for j in range(batchsize):
        hr_image.append(np.asarray(batch[j][0]).astype("f"))
        lr_image.append(np.asarray(batch[j][1]).astype("f"))
        label.append(np.asarray(batch[j][2]).astype(np.int32))
    hr_image = xp.asarray(hr_image)
    lr_image = xp.asarray(lr_image)
    label = xp.asarray(label)
    return hr_image, lr_image, label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/base.yml')
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--data_dir_eval', type=str, default='')
    parser.add_argument('--results_dir', type=str, default='./results/gans')
    parser.add_argument('--snapshot', type=str, default='')
    parser.add_argument('--inception_model_path', type=str,
                        default='')
    parser.add_argument('--n_images', type=int, default=5000)
    parser.add_argument('--basic_sr', type=str, default=None)
    parser.add_argument('--tf', action='store_true', default=False)
    args = parser.parse_args()
    chainer.cuda.get_device_from_id(args.gpu).use()
    config = yaml_utils.Config(yaml.load(open(args.config_path)))
    # Model
    gen = load_models(config)
    gen.to_gpu()
    chainer.serializers.load_npz(args.snapshot, gen)
    np.random.seed(1234)
    xp = gen.xp
    ims = []
    cs = []
    config['dataset_eval']['args']['root'] = args.data_dir_eval
    config['dataset_eval']['args']['use_label'] = True
    dataset_eval = load_dataset_eval(config)
    bs = 100
    eval_iter = chainer.iterators.SerialIterator(dataset_eval, bs, shuffle=False)
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        for _ in range(0, args.n_images, bs):
            hr_image, lr_image, label = get_batch_sr_with_label(eval_iter, xp)
            if args.basic_sr is not None:
                lr_image = F.clip(lr_image * 127.5 + 127.5, 0.0, 255.0)
                lr_image = cuda.to_cpu(lr_image.data)
                n_img = []
                for img in lr_image:
                    n_img.append(imresize(img.transpose(1, 2, 0), (299, 299), args.basic_sr).transpose(2, 0, 1))
                lr_image = np.array(n_img, np.float32)
                ims.append(cuda.to_cpu(lr_image))
            else:
                hr_image_fake = gen(lr_image)
                hr_image_fake = F.clip(hr_image_fake * 127.5 + 127.5, 0.0, 255.0)
                ims.append(cuda.to_cpu(hr_image_fake.data))
            cs.append(cuda.to_cpu(label))
    ims = np.concatenate(ims, 0)
    cs = np.concatenate(cs, 0)
    mapping = np.loadtxt('datasets/inception_class_map_info/ids.txt', dtype=int)
    mapping = list(mapping)
    cs = np.array([mapping[c] for c in cs], np.int32)

    if args.tf:
        import source.inception.inception_score_tf
        from source.inception.inception_score_tf import get_inception_accuracy
        acc = get_inception_accuracy(ims, cs)
        print(acc)
    else:
        from source.inception.inception_score import inception_accuracy
        model = load_inception_model(args.inception_model_path)
        acc = inception_accuracy(model, ims, cs, splits=1)
        print(acc)
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    print("Inception accuracy:", acc)
    np.savetxt('{}/inception_accuracy.txt'.format(args.results_dir), np.array([acc]))


if __name__ == '__main__':
    main()
