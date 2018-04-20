import numpy as np
import chainer


def sample_continuous(dim, batchsize, distribution='normal', xp=np):
    if distribution == "normal":
        return xp.random.randn(batchsize, dim) \
            .astype(xp.float32)
    elif distribution == "uniform":
        return xp.random.uniform(-1, 1, (batchsize, dim)) \
            .astype(xp.float32)
    else:
        raise NotImplementedError


def sample_categorical(n_cat, batchsize, distribution='uniform', xp=np):
    if distribution == 'uniform':
        return xp.random.randint(low=0, high=n_cat, size=(batchsize)).astype(xp.int32)
    else:
        raise NotImplementedError


def sample_from_categorical_distribution(batch_probs):
    """Sample a batch of actions from a batch of action probabilities.
    Args:
        batch_probs (ndarray): batch of action probabilities BxA
    Returns:
        ndarray consisting of sampled action indices
    """
    xp = chainer.cuda.get_array_module(batch_probs)
    return xp.argmax(
        xp.log(batch_probs) + xp.random.gumbel(size=batch_probs.shape),
        axis=1).astype(np.int32, copy=False)
