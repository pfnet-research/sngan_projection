import os, sys
import numpy as np
import argparse
import chainer

base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base, '../'))
from evaluation import gen_images
import yaml
import source.yaml_utils as yaml_utils


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
    parser.add_argument('--inception_model_path', type=str, default='')
    parser.add_argument('--splits', type=int, default=10)
    parser.add_argument('--tf', action='store_true', default=False)
    args = parser.parse_args()
    chainer.cuda.get_device_from_id(args.gpu).use()
    config = yaml_utils.Config(yaml.load(open(args.config_path)))
    # Model
    gen = load_models(config)
    gen.to_gpu(args.gpu)
    chainer.serializers.load_npz(args.snapshot, gen)
    np.random.seed(1234)
    xp = gen.xp
    n = int(5000 * args.splits)
    #for _ in range(50):
    #     gen(128) 
    print("Gen")
    ims = gen_images(gen, n, batchsize=100).astype("f")
    print(np.max(ims), np.min(ims))

    if args.tf:
        import source.inception.inception_score_tf as inception_score
        mean, std = inception_score.get_inception_score(ims, args.splits)
        print(mean, std)
    else:
        from evaluation import load_inception_model
        from source.inception.inception_score import inception_score, Inception
        model = load_inception_model(args.inception_model_path)
        mean, std = inception_score(model, ims, splits=args.splits)
        print(mean, std)
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    np.savetxt('{}/inception_score.txt'.format(args.results_dir),
               np.array([mean, std]))


if __name__ == '__main__':
    main()
