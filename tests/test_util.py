import itertools as it
import numpy as np
import pytest
from scipy import stats
import summaries


NUM_SAMPLES = 10000


@pytest.mark.parametrize('size', [(NUM_SAMPLES, 1), NUM_SAMPLES])
def test_estimate_entropy(size):
    scale = 7
    x = np.random.normal(0, scale, size)
    actual = summaries.estimate_entropy(x)
    expected = (np.log(2 * np.pi * scale ** 2) + 1) / 2
    assert abs(actual - expected) < .1


@pytest.mark.parametrize('method, normalize', it.product(['singh', 'kl'], ['x', 'y', 'xy', False]))
def test_estimate_mutual_information(method, normalize):
    cov = np.asarray([[1, .7], [.7, 2]])
    x = np.random.multivariate_normal(np.zeros(2), cov, size=NUM_SAMPLES)
    actual = summaries.estimate_mutual_information(*x.T, method=method, normalize=normalize)
    expected = (np.log(np.diag(cov)).sum() - np.linalg.slogdet(cov)[1]) / 2
    if normalize == 'x':
        norm = stats.norm(0, cov[0, 0]).entropy()
    elif normalize == 'y':
        norm = stats.norm(0, cov[1, 1]).entropy()
    elif normalize == 'xy':
        norm = stats.norm(0, np.diag(cov)).entropy().mean()
    elif not normalize:
        norm = 1
    else:
        raise NotImplementedError(normalize)
    expected /= norm
    assert abs(actual - expected) < .1


def test_rmse_and_rmse_uniform():
    x, y = np.random.uniform(0, 2, (2, NUM_SAMPLES))
    actual = summaries.evaluate_rmse(x, y)
    expected = summaries.evaluate_rmse_uniform(2)
    assert abs(actual - expected) < .1


def test_mae_and_mae_uniform():
    x, y = np.random.uniform(0, 2, (2, NUM_SAMPLES))
    actual = summaries.evaluate_mae(x, y)
    expected = summaries.evaluate_mae_uniform(2)
    assert abs(actual - expected) < .1
