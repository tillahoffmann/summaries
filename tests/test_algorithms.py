import numpy as np
import pytest
from scipy import stats
from sklearn.exceptions import NotFittedError
from summaries.algorithms import NearestNeighborAlgorithm
from summaries.base import Container, ParamDict


def test_nearest_neighbor_algorithm(basic_data: Container[np.ndarray],
                                    basic_params: Container[ParamDict]) -> None:
    # Fit the nearest neighbor algorithm.
    frac = 0.01
    algorithm = NearestNeighborAlgorithm(frac)
    algorithm.fit(basic_data, basic_params)

    # Sample from the approximate posterior and verify its shape.
    samples = algorithm.predict(basic_data.observed)
    expected_shape = (basic_data.observed.shape[0], int(frac * basic_data.simulated.shape[0]))
    assert samples["theta"].shape == expected_shape
    assert samples["dummy"].shape == expected_shape

    # Verify that the posterior mean is predictive of the parameter.
    pearsonr = stats.pearsonr(basic_params.observed["theta"], samples["theta"].mean(axis=-1))
    assert pearsonr.statistic > 0.9 and pearsonr.pvalue < 1e-3

    # Verify that we have not learned anything about the dummy parameter.
    pearsonr = stats.pearsonr(basic_params.observed["dummy"], samples["dummy"].mean(axis=-1))
    assert pearsonr.statistic < 0.5 and pearsonr.pvalue > 1e-3


def test_nearest_neighbor_algorithm_invalid():
    algorithm = NearestNeighborAlgorithm(0.01)
    with pytest.raises(NotFittedError):
        algorithm.predict(None)

    algorithm.fit(np.random.normal(0, 1, (10, 3)), None)
    with pytest.raises(ValueError, match="must be a matrix"):
        algorithm.predict(0)
    with pytest.raises(ValueError, match="must have 3 features"):
        algorithm.predict(np.random.normal(0, 1, (3, 4)))
