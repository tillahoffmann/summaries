import numpy as np
import pytest
import summaries


@pytest.fixture(params=[False, True])
def configuration(request: pytest.FixtureRequest):
    n = 10_000
    mu = np.random.normal(0, 1, (n, 1))
    x = np.random.normal(mu, 1, (n, 10)).mean(axis=1, keepdims=True)
    return {
        "algorithm": summaries.NearestNeighborAlgorithm(x, mu, standardize=request.param),
        "x": x,
        "mu": mu,
    }


def test_rejection_algorithm(configuration: dict) -> None:
    x = configuration["x"]
    mu = configuration["mu"]
    algorithm = configuration["algorithm"]

    # Sample from the approximate posterior.
    num_samples = 100
    y, info = algorithm.sample(x[0], num_samples + 1)

    # Drop the first sample (because it's what we used to generate the data).
    assert info['distances'][0, 0] == 0
    i = info['indices'][0, 1:]
    y = y[0, 1:]

    # Verify shape and ensure MSE of the sample is smaller than MSE for the entire reference table.
    assert i.shape == (num_samples,)
    mse_all = np.square(mu[0] - mu[1:]).mean()
    mse_sample = np.square(mu[0] - y).mean()
    assert mse_all > mse_sample

    assert algorithm.num_params == 1


def test_rejection_algorithm_batch_query(configuration: dict):
    batch_size = 13
    num_samples = 17
    x = configuration["x"][:batch_size]
    y, info = configuration["algorithm"].sample(x, num_samples)
    assert y.shape == (batch_size, num_samples, configuration["algorithm"].num_params)
    assert info['indices'].shape == (batch_size, num_samples)
    assert info['distances'].shape == (batch_size, num_samples)


def test_logger(configuration: summaries.Algorithm):
    algorithm = configuration["algorithm"]
    algorithm.logger.debug('logging test for %s', algorithm)
