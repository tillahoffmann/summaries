import numpy as np
import pytest
import summaries


@pytest.fixture
def algorithm(request):
    n = 10000
    mu = np.random.normal(0, 1, (n, 1))
    x = np.random.normal(mu, 1, (n, 10)).mean(axis=1, keepdims=True)
    return summaries.NearestNeighborAlgorithm(x, mu)


def test_rejection_algorithm(algorithm: summaries.NearestNeighborAlgorithm):
    x = algorithm.train_data
    mu = algorithm.train_params

    # Sample from the approximate posterior.
    num_samples = 100
    y, info = algorithm.sample(x[0], num_samples + 1)

    # Drop the first sample (because it's what we used to generate the data).
    assert info['distances'][0] == 0
    i = info['indices'][1:]
    y = y[1:]

    # Verify shape and ensure MSE of the sample is smaller than MSE for the entire reference table.
    assert i.shape == (num_samples,)
    mse_all = np.square(mu[0] - mu[1:]).mean()
    mse_sample = np.square(mu[0] - y).mean()
    assert mse_all > mse_sample

    assert algorithm.num_params == 1


def test_rejection_algorithm_batch_query(algorithm: summaries.NearestNeighborAlgorithm):
    batch_size = 13
    num_samples = 17
    x = algorithm.train_data[:batch_size]
    y, info = algorithm.sample(x, num_samples)
    assert y.shape == (batch_size, num_samples, algorithm.num_params)
    assert info['indices'].shape == (batch_size, num_samples)
    assert info['distances'].shape == (batch_size, num_samples)


def test_logger(algorithm: summaries.Algorithm):
    algorithm.logger.debug('logging test for %s', algorithm)
