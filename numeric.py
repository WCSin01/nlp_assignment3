import numpy as np
from scipy.special import logsumexp


def log_mat_mul(logA: np.ndarray, logB: np.ndarray) -> np.ndarray:
    """

    :param logA:
    :param logB:
    :return: log(AB). AB is matrix multiplication
    """
    m, n = logA.shape
    p = logB.shape[1]
    assert logB.shape == (n, p)
    log_pairwise_products = \
        np.broadcast_to(np.expand_dims(logA, 2), (m, n, p)) + np.broadcast_to(np.expand_dims(logB, 0), (m, n, p))
    return logsumexp(log_pairwise_products, axis=1)


def log_normalize(log_array: np.ndarray, axis=None) -> np.ndarray:
    """
    normalize in log space

    :param log_array:
    :param axis:
    :return: normalized array no longer in log space
    """
    # max_ = np.expand_dims(np.max(log_array, axis=axis), axis=axis)
    # array = np.exp(log_array - np.ma.array(max_, mask=np.isinf(max_)))
    # sum_ = np.sum(array, axis=axis)
    # array = array / np.expand_dims(sum_, axis=axis)
    # return array.filled(fill_value=0)

    log_sum = np.expand_dims(logsumexp(log_array, axis=axis), axis=axis)
    log_array = log_array - np.ma.array(log_sum, mask=(np.isinf(log_sum)))
    return log_array.filled(fill_value=-np.inf)


def masked_normalize(array: np.ndarray, axis=None) -> np.ndarray:
    sum_ = np.sum(array, axis=axis)
    array = array / np.expand_dims(np.ma.array(sum_, mask=(sum_==0)), axis=axis)
    return array.filled(fill_value=0)
