# Code derived from https://github.com/openai/improved-gan/tree/master/inception_score
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
import glob
import scipy.misc
import math
import sys
import chainer
from chainer import functions as F

MODEL_DIR = '/tmp/imagenet'
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
softmax = None
last_layer = None

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3


def inception_forward(images, layer):
    assert (type(images[0]) == np.ndarray)
    assert (len(images[0].shape) == 3)
    assert (np.max(images[0]) > 10)
    assert (np.min(images[0]) >= 0.0)
    bs = 100
    images = images.transpose(0, 2, 3, 1)
    with tf.Session(config=config) as sess:
        preds = []
        n_batches = int(math.ceil(float(len(images)) / float(bs)))
        for i in range(n_batches):
            sys.stdout.write(".")
            sys.stdout.flush()
            inp = images[(i * bs):min((i + 1) * bs, len(images))]
            pred = sess.run(layer, {'ExpandDims:0': inp})
            preds.append(pred)
        preds = np.concatenate(preds, 0)
    return preds


def get_mean_and_cov(images):
    before_preds = inception_forward(images, last_layer)
    m = np.mean(before_preds, 0)
    cov = np.cov(before_preds, rowvar=False)
    return m, cov


def get_fid(images, ref_stats=None, images_ref=None, splits=10):
    before_preds = inception_forward(images, last_layer)
    if ref_stats is None:
        if images_ref is None:
            raise ValueError('images_ref should be provided if ref_stats is None')
        m_ref, cov_ref = get_mean_and_cov(images_ref)
    fids = []
    for i in range(splits):
        part = before_preds[(i * before_preds.shape[0] // splits):((i + 1) * before_preds.shape[0] // splits), :]
        m_gen = np.mean(part, 0)
        cov_gen = np.cov(part, rowvar=False)
        fid = np.sum((m_ref - m_gen) ** 2) + np.trace(
            cov_ref + cov_gen - 2 * scipy.linalg.sqrtm(np.dot(cov_ref, cov_gen)))
        fids.append(fid)
    return np.mean(fids), np.std(fids)


# Call this function with list of images. Each of elements should be a
# numpy array with values ranging from 0 to 255.
def get_inception_score(images, splits=10):
    preds = inception_forward(images, softmax)
    scores = []
    for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)


# Call this function with list of images. Each of elements should be a
# numpy array with values ranging from 0 to 255.
def get_inception_accuracy(images, labels):
    batch_size = 100
    if isinstance(images, (list, tuple)):
        ims_list = images
        ys_list = []
        for ims in ims_list:
            n, _, _, _ = ims.shape
            n_batches = int(math.ceil(float(n) / float(batch_size)))
            print('batch_size:{}, n_ims{}, n_batches{}'.format(batch_size, n, n_batches))
            print('Calculating inception accuracy...')
            ys = inception_forward(ims, softmax)[:, 1:1001]
            ys_list.append(ys)
        ys = sum(ys_list) / len(ys_list)
    else:
        n, _, _, _, = images.shape
        n_batches = int(math.ceil(float(n) / float(batch_size)))

        print('batch_size:{}, n_ims{}, n_batches{}'.format(batch_size, n, n_batches))
        print('Calculating inception accuracy...')
        ys = inception_forward(images, softmax)[:, 1:1001]
    return F.accuracy(ys, labels).data


# This function is called automatically.
def _init_inception():
    global softmax
    global last_layer
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)
    with tf.gfile.FastGFile(os.path.join(
            MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
    # Works with an arbitrary minibatch size.
    with tf.Session(config=config) as sess:
        pool3 = sess.graph.get_tensor_by_name('pool_3:0')
        ops = pool3.graph.get_operations()
        for op_idx, op in enumerate(ops):
            for o in op.outputs:
                shape = o.get_shape()
                shape = [s.value for s in shape]
                new_shape = []
                for j, s in enumerate(shape):
                    if s == 1 and j == 0:
                        new_shape.append(None)
                    else:
                        new_shape.append(s)
                o._shape = tf.TensorShape(new_shape)
        w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
        last_layer = tf.squeeze(pool3)
        logits = tf.matmul(last_layer, w)
        softmax = tf.nn.softmax(logits)


if softmax is None:
    _init_inception()
