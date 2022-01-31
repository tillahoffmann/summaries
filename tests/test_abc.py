import numpy as np
import pytest
from scipy import spatial
import summaries


@pytest.mark.parametrize('use_kdtree, size', [
    (False, 10000),
    (True, (10000, 1)),
])
def test_abc(use_kdtree: bool, size: tuple):
    # Generate data from a toy model.
    mu = np.random.normal(0, 1, size)
    x = np.random.normal(mu, 1, size)

    # Construct the reference table/tree.
    reference = spatial.KDTree(x) if use_kdtree else x

    # Sample from the approximate posterior.
    num_samples = 999
    d, i = summaries.sample_posterior(x[0], reference, num_samples + 1)
    # Drop the first sample (because it's what we used to generate the data).
    assert d[0] == 0
    i = i[1:]

    # Verify shape and ensure MSE of the sample is smaller than MSE for the entire reference table.
    assert i.shape == (num_samples,)
    mse_all = np.square(mu[0] - mu).mean()
    mse_sample = np.square(mu[0] - mu[i]).mean()
    assert mse_all > mse_sample
