import numpy as np
import pytest
import summaries


@pytest.mark.parametrize('size, whiten', [
    (10000, False),
    ((10000, 1), False),
    ((10000, 1), True),
])
def test_abc(size: tuple, whiten: bool):
    # Generate data from a toy model.
    mu = np.random.normal(0, 1, size)
    x = np.random.normal(mu, 1, size)

    # Construct the reference table/tree.
    algorithm = summaries.RejectionAlgorithm(x, whiten)

    # Sample from the approximate posterior.
    num_samples = 99
    i, d = algorithm.sample_posterior(x[0], num_samples + 1, True)
    # Drop the first sample (because it's what we used to generate the data).
    assert d[0] == 0
    i = i[1:]

    # Verify shape and ensure MSE of the sample is smaller than MSE for the entire reference table.
    y = mu[i]
    assert i.shape == (num_samples,)
    mse_all = np.square(mu[0] - mu[1:]).mean()
    mse_sample = np.square(mu[0] - y).mean()
    assert mse_all > mse_sample
