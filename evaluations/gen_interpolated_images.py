import os, sys, time
import shutil
import numpy as np
import argparse
import chainer
from PIL import Image

base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base, '../'))
from evaluation import gen_images_with_condition
import yaml
import source.yaml_utils as yaml_utils
from source.miscs.random_samples import sample_gaussian


def load_models(config):
    gen_conf = config.models['generator']
    gen = yaml_utils.load_model(gen_conf['fn'], gen_conf['name'], gen_conf['args'])
    return gen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/base.yml')
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--results_dir', type=str, default='./results/gans')
    parser.add_argument('--snapshot', type=str, default='')
    parser.add_argument('--n_ys', type=int, default=5)
    parser.add_argument('--n_zs', type=int, default=5)
    parser.add_argument('--classes', type=int, nargs="*", default=None)
    parser.add_argument('--n_images', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()
    chainer.cuda.get_device(args.gpu).use()
    config = yaml_utils.Config(yaml.load(open(args.config_path)))
    # Model
    gen = load_models(config)
    gen.to_gpu()
    out = args.results_dir
    chainer.serializers.load_npz(args.snapshot, gen)
    np.random.seed(args.seed)
    xp = gen.xp
    n_images = args.n_zs * args.n_ys

    for k in range(args.n_images):
        print(k)
        classes = tuple(args.classes) if args.classes is not None else [np.random.randint(1000),
                                                                        np.random.randint(1000)]
        classes = [xp.array([classes[0]] * n_images, xp.int32), xp.array([classes[1]] * n_images, xp.int32)]
        noises = [np.random.normal(size=(128,)), np.random.normal(size=(128,))]
        noises = [xp.array([noises[0]] * n_images, dtype=xp.float32),
                  xp.array([noises[1]] * n_images, dtype=xp.float32)]
        ws_z_0 = []
        ws_z_1 = []
        ws_y_0 = []
        ws_y_1 = []
        for i in np.linspace(0, 1, args.n_zs):
            for j in np.linspace(0, 1, args.n_ys):
                ws_z_0.append(1 - i)
                ws_z_1.append(i)
                ws_y_0.append(1 - j)
                ws_y_1.append(j)
        ws_z = [xp.array(ws_z_0, xp.float32), xp.array(ws_z_1, xp.float32)]
        ws_y = [xp.array(ws_y_0, xp.float32), xp.array(ws_y_1, xp.float32)]
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            x = gen(zs=noises, ys=classes, ws_z=ws_z, ws_y=ws_y)
        x = chainer.cuda.to_cpu(x.data)
        x = np.asarray(np.clip(x * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
        print(x.shape)
        _, _, h, w = x.shape
        x = x.reshape((args.n_zs, args.n_ys, 3, h, w))
        x = x.transpose(0, 3, 1, 4, 2)
        x = x.reshape((args.n_zs * h, args.n_ys * w, 3))

        save_path = os.path.join(out, 'interpolated_images_{}.png'.format(k))
        if not os.path.exists(out):
            os.makedirs(out)
        Image.fromarray(x).save(save_path)


if __name__ == '__main__':
    main()
