import numpy as np
from scipy import stats
from summaries import benchmark
import torch as th


def test_benchmark_coverage():
    """
    This test verifies that the exact posterior (based on numerical integration) has correct
    coverage in the sense that the parameter value used to generate synthetic value has
    approximately uniform rank within the posteriors.
    """
    eps = 1e-6
    lin = th.linspace(-1 + eps, 1 - eps, 1000)
    dx = lin[1] - lin[0]
    quantiles = []
    for _ in range(100):
        # Sample and evaluate the posterior.
        data = benchmark.sample()
        log_posterior = benchmark.evaluate_log_joint(data['x'], lin)
        posterior = np.exp(log_posterior)

        # Evaluate the quantile of the true value.
        fltr = lin < data["theta"]
        quantile = th.trapz(posterior[fltr], dx=dx).item()
        quantiles.append(quantile)

    ks = stats.ks_1samp(quantiles, lambda x: x)
    assert ks.pvalue > 0.001


def test_benchmark_stan_model():
    # Generate some data and fit the model.
    data = benchmark.sample()
    xs = data['x'][..., :1]
    model = benchmark.StanBenchmarkAlgorithm('summaries/benchmark.stan')
    samples, info = model.sample(np.asarray([xs.numpy()] * 3), 1000, keep_fits=True)

    # Validate the output and ensure the likelihood is the same in python and stan.
    assert samples.shape == (3, 1000, 1)
    fit = info['fits'][0]
    variables = fit.stan_variables()
    likelihood = benchmark.evaluate_gaussian_mixture_distribution(th.as_tensor(variables['theta']))
    # Make sure the parameters of the likelihood are the same.
    component_dist: th.distributions.Normal = likelihood._component_distribution
    np.testing.assert_allclose(variables['theta_'], component_dist.loc[:, 0].abs())
    np.testing.assert_allclose(variables['scale'], component_dist.scale[:, 0])
    # Compare the likelihoods for each value.
    for x, part in zip(xs, variables['target_parts'].T):
        log_prob = likelihood.log_prob(x)
        np.testing.assert_allclose(part, log_prob, rtol=1e-4)


def test_benchmark_batch_sample():
    batch_size = 47
    batch = benchmark.sample(size=batch_size)
    assert batch['theta'].shape == (batch_size, 1)
    assert batch['x'].shape == (batch_size, benchmark.NUM_OBSERVATIONS,
                                1 + benchmark.NUM_NOISE_FEATURES)


def preprocess_candidate_features():
    batch = benchmark.sample(size=47)
    features = benchmark.preprocess_candidate_features(batch['x'])
    assert features.shape == (47, 6)
