import numpy as np
import pytest
from scipy import stats
from summaries import distributions
import typing


@pytest.fixture(params=[None, 4, (3, 7)])
def batch_shape(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture(params=['uniform', 'negative_binomial', 'normal'])
def distribution_pair(request: pytest.FixtureRequest, batch_shape):
    if request.param == 'uniform':
        lower = np.random.uniform(0, 1, batch_shape)
        width = np.random.uniform(0, 1, batch_shape)
        dist = distributions.UniformDistribution(lower, lower + width)
        scipy_dist = None if batch_shape else stats.uniform(lower, width)
    elif request.param == 'negative_binomial':
        n = np.random.poisson(10, batch_shape)
        p = np.random.uniform(0, 1, batch_shape)
        dist = distributions.NegativeBinomialDistribution(n, p)
        scipy_dist = None if batch_shape else stats.nbinom(n, p)
    elif request.param == 'normal':
        loc = np.random.normal(0, 1, 5)
        scale = np.random.gamma(10, .1, 5)
        dist = distributions.NormalDistribution(loc, scale)
        scipy_dist = None if batch_shape else stats.norm(loc, scale)
    else:
        raise NotImplementedError
    return dist, scipy_dist


def test_scipy_equivalence(distribution_pair: typing.Tuple):
    dist, scipy_dist = distribution_pair
    if scipy_dist is None:
        pytest.skip()
    x = dist.sample()
    try:
        expected_log_prob = scipy_dist.logpmf(x)
    except AttributeError:
        expected_log_prob = scipy_dist.logpdf(x)
    np.testing.assert_allclose(expected_log_prob, dist.log_prob(x))
    np.testing.assert_allclose(
        scipy_dist.mean() if callable(scipy_dist.mean) else scipy_dist.mean, dist.mean
    )


@pytest.mark.parametrize('size', [None, 6, (9, 13)])
def test_batch_sample(size, distribution_pair):
    dist: distributions.Distribution
    dist, _ = distribution_pair
    x = dist.sample(size)
    expected_shape = dist._normalize_shape(size)
    assert np.shape(x)[:len(expected_shape)] == expected_shape
