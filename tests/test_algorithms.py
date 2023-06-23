import numpy as np
import pytest
from scipy import stats
from sklearn.exceptions import NotFittedError
from summaries.algorithms import NearestNeighborAlgorithm


def test_nearest_neighbor_algorithm():
    # Generate some data.
    n = 100_000
    m = 100
    theta = np.random.normal(0, 1, n + m)
    x = theta[:, None] + np.random.normal(0, 0.1, (n + m, 2))

    # Fit the nearest neighbor algorithm.
    frac = 0.01
    algorithm = NearestNeighborAlgorithm(frac)
    algorithm.fit(x[:n], {"theta": theta[:n]})

    # Sample from the approximate posterior and verify that the posterior mean is predictive of the
    # parameter.
    samples = algorithm.predict(x[n:])
    pearsonr = stats.pearsonr(theta[n:], samples["theta"].mean(axis=-1))
    assert pearsonr.statistic > 0.9 and pearsonr.pvalue < 1e-3
    assert samples["theta"].shape == (m, int(frac * n),)


def test_nearest_neighbor_algorithm_invalid():
    algorithm = NearestNeighborAlgorithm(0.01)
    with pytest.raises(NotFittedError):
        algorithm.predict(None)

    algorithm.fit(np.random.normal(0, 1, (10, 3)), None)
    with pytest.raises(ValueError, match="must be a matrix"):
        algorithm.predict(0)
    with pytest.raises(ValueError, match="must have 3 features"):
        algorithm.predict(np.random.normal(0, 1, (3, 4)))
