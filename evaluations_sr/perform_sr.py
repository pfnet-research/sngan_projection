import os, sys, time
import shutil
import numpy as np
import argparse
import chainer
from chainer import functions as F
from PIL import Image
import yaml

base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base, '../'))
import source.yaml_utils as yaml_utils
from extentions_for_eval import get_batch
import scipy.misc


def load_models(config):
    gen_conf = config.models['generator']
    gen = yaml_utils.load_model(gen_conf['fn'], gen_conf['name'], gen_conf['args'])
    return gen


def load_dataset_eval(config):
    dataset = yaml_utils.load_module(config.dataset_eval['dataset_fn'],
                                     config.dataset_eval['dataset_name'])
    return dataset(**config.dataset_eval['args'])


def array2img(array, xp):
    array = xp.clip(array * 127.5 + 127.5, 0.0, 255.0)
    return chainer.cuda.to_cpu(array).transpose(0, 2, 3, 1).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/base.yml')
    parser.add_argument('--data_dir_eval', type=str, default='')
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--results_dir', type=str, default='./results_dir')
    parser.add_argument('--snapshot', type=str, default='')
    parser.add_argument('--n_images', type=int, default=100)
    args = parser.parse_args()
    chainer.cuda.get_device(args.gpu).use()
    config = yaml_utils.Config(yaml.load(open(args.config_path)))
    # Model
    gen = load_models(config)
    gen.to_gpu()
    out = args.results_dir
    config['dataset_eval']['args']['root'] = args.data_dir_eval
    dataset_eval = load_dataset_eval(config)
    eval_iter = chainer.iterators.SerialIterator(dataset_eval, 10, shuffle=False)
    if not os.path.exists(out):
        os.makedirs(out)
    chainer.serializers.load_npz(args.snapshot, gen)
    np.random.seed(1234)
    xp = gen.xp
    ims = []
    ims_org = []
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        eval_iter.reset()
        for _ in range(0, args.n_images, 10):
            print(_)
            _, lr_image = get_batch(eval_iter, xp)
            hr_image_fake = gen(lr_image).data
            ims.append(array2img(hr_image_fake, xp))
            ims_org.append(array2img(lr_image, xp))
            del hr_image_fake, lr_image
    ims = np.concatenate(ims, 0)
    ims_org = np.concatenate(ims_org, 0)
    print(ims.shape)
    if not os.path.exists(out):
        os.makedirs(out)
    ims_dir_path = os.path.join(out, 'images_hr')
    if not os.path.exists(ims_dir_path):
        os.makedirs(ims_dir_path)
    ims_dir_path = os.path.join(out, 'images_lr')
    if not os.path.exists(ims_dir_path):
        os.makedirs(ims_dir_path)
    ims_dir_path = os.path.join(out, 'images_hr_bicubic')
    if not os.path.exists(ims_dir_path):
        os.makedirs(ims_dir_path)
    for i, (im, im_org) in enumerate(zip(ims, ims_org)):
        print(i)
        print(im.shape)
        save_path = os.path.join('images_hr', '{}.png'.format(str(i)))
        Image.fromarray(im).save(os.path.join(out, save_path))
        save_path = os.path.join('images_lr', '{}.png'.format(str(i)))
        Image.fromarray(im_org).save(os.path.join(out, save_path))
        im_bicubic = scipy.misc.imresize(im_org, [im.shape[0], im.shape[1]], 'bicubic')
        save_path = os.path.join('images_hr_bicubic', '{}.png'.format(str(i)))
        Image.fromarray(im_bicubic).save(os.path.join(out, save_path))


if __name__ == '__main__':
    main()
