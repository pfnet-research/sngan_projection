import os, sys
import numpy as np
import argparse
import chainer

base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base, '../'))
from evaluation import gen_images, gen_images_with_condition, load_inception_model
import yaml
import source.yaml_utils as yaml_utils
from evaluation import FID


def load_models(config):
    gen_conf = config.models['generator']
    gen = yaml_utils.load_model(gen_conf['fn'], gen_conf['name'], gen_conf['args'])
    return gen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/base.yml')
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--results_dir', type=str, default='./results/gans')
    parser.add_argument('--stat_dir_path', type=str,
                        default='')
    parser.add_argument('--inception_model_path', type=str,
                        default='')
    parser.add_argument('--snapshot', type=str, default='')
    parser.add_argument('--n_classes', type=int, default=1000)
    parser.add_argument('--class_start_from', type=int, default=0)
    parser.add_argument('--tf', action='store_true', default=False)
    args = parser.parse_args()
    chainer.cuda.get_device_from_id(args.gpu).use()
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    config = yaml_utils.Config(yaml.load(open(args.config_path)))
    # Model
    gen = load_models(config)
    gen.to_gpu()
    chainer.serializers.load_npz(args.snapshot, gen)
    np.random.seed(1234)
    if args.tf:
        import source.inception.inception_score_tf
        from source.inception.inception_score_tf import get_mean_and_cov as get_mean_cov
    else:
        from evaluation import get_mean_cov
        model = load_inception_model(args.inception_model_path)
    for c in range(args.class_start_from, args.n_classes):
        print("class:{}".format(c))
        stat = np.load(os.path.join(args.stat_dir_path, '{}.npz'.format(c)))
        ims = gen_images_with_condition(gen, c, 5000, batchsize=100).astype("f")
        if args.tf:
            mean, cov = get_mean_cov(ims)
        else:
            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                mean, cov = get_mean_cov(model, ims, batch_size=100)
        fid = FID(stat["mean"], stat["cov"], mean, cov)
        print(fid)
        np.savetxt('{}/fid_{}.txt'.format(args.results_dir, c), np.array([fid]))


if __name__ == '__main__':
    main()
