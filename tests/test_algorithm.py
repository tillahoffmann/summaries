import numpy as np
import pytest
import summaries


@pytest.mark.parametrize('whiten', [False, True])
def test_abc(whiten: bool):
    # Generate data from a toy model.
    n = 10000
    mu = np.random.normal(0, 1, (n, 1))
    x = np.random.normal(mu, 1, (n, 10)).mean(axis=1, keepdims=True)

    # Construct the reference table/tree.
    algorithm = summaries.RejectionAlgorithm(x, mu, whiten)

    # Sample from the approximate posterior.
    num_samples = 99
    y1 = algorithm.sample_posterior(x[0], num_samples + 1)
    y, i, d = algorithm.sample_posterior(x[0], num_samples + 1, True, True)
    np.testing.assert_array_equal(y, y1)
    # Drop the first sample (because it's what we used to generate the data).
    assert d[0] == 0
    i = i[1:]
    y = y[1:]

    # Verify shape and ensure MSE of the sample is smaller than MSE for the entire reference table.
    assert i.shape == (num_samples,)
    mse_all = np.square(mu[0] - mu[1:]).mean()
    mse_sample = np.square(mu[0] - y).mean()
    assert mse_all > mse_sample

    assert algorithm.num_params == 1
