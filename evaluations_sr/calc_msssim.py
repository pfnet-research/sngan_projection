import os, sys
import numpy as np
import argparse
import chainer
from chainer import functions as F
from chainer import cuda
import yaml
from scipy.misc import imresize

base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base, '../'))
import source.yaml_utils as yaml_utils
from source.functions.msssim import RGBMSSSIM
from extentions_for_eval import sample_generate, sample_generate_light, get_batch


def load_models(config):
    gen_conf = config.models['generator']
    gen = yaml_utils.load_model(gen_conf['fn'], gen_conf['name'], gen_conf['args'])
    return gen


def load_dataset_eval(config):
    dataset = yaml_utils.load_module(config.dataset_eval['dataset_fn'],
                                     config.dataset_eval['dataset_name'])
    return dataset(**config.dataset_eval['args'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/base.yml')
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--data_dir_eval', type=str, default='')
    parser.add_argument('--results_dir', type=str, default='./results/gans')
    parser.add_argument('--snapshot', type=str, default='')
    parser.add_argument('--n_images', type=int, default=1000)
    parser.add_argument('--basic_sr', type=str, default=None)
    args = parser.parse_args()
    chainer.cuda.get_device(args.gpu).use()
    config = yaml_utils.Config(yaml.load(open(args.config_path)))
    # Model
    gen = load_models(config)
    gen.to_gpu()
    config['dataset_eval']['args']['root'] = args.data_dir_eval
    dataset_eval = load_dataset_eval(config)
    eval_iter = chainer.iterators.SerialIterator(dataset_eval, 10, shuffle=False)
    out = args.results_dir
    if not os.path.exists(out):
        os.makedirs(out)
    chainer.serializers.load_npz(args.snapshot, gen)
    np.random.seed(1234)
    xp = gen.xp
    msssims = []
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        for _ in range(0, args.n_images, 10):
            hr_image, lr_image = get_batch(eval_iter, xp)
            if args.basic_sr is not None:
                lr_image = F.clip(lr_image * 127.5 + 127.5, 0.0, 255.0)
                lr_image = cuda.to_cpu(lr_image.data)
                n_img = []
                for img in lr_image:
                    n_img.append(imresize(img.transpose(1, 2, 0), (hr_image.shape[2], hr_image.shape[3]),
                                          args.basic_sr).transpose(2, 0, 1))
                hr_image_fake = chainer.Variable(xp.array(n_img, np.float32))
            else:
                hr_image_fake = gen(lr_image)
                hr_image_fake = F.clip(hr_image_fake * 127.5 + 127.5, 0.0, 255.0)
            hr_image = F.resize_images(F.clip(hr_image * 127.5 + 127.5, 0.0, 255.0), (256, 256))
            hr_image_fake = F.resize_images(hr_image_fake, (256, 256))
            msssim = F.mean(RGBMSSSIM(max_val=255)(hr_image, hr_image_fake)).data
            msssims.append(msssim)
            del lr_image, hr_image, hr_image_fake
    print('Average M3SIM:', sum(msssims) / len(msssims))
    np.savetxt('{}/msssim.txt'.format(out), np.array([sum(msssims) / len(msssims)]))


if __name__ == '__main__':
    main()
