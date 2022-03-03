import numpy as np
from scipy import stats
import summaries
from summaries import benchmark
import torch as th


def test_benchmark_coverage():
    """
    This test verifies that the exact posterior (based on numerical integration) has correct
    coverage in the sense that the `1 - alpha` credible interval contains the true value in
    `1 - alpha` of the simulated cases.
    """
    alpha = .3
    lin = th.linspace(-4, 4, 200)
    levels = []
    for _ in range(100):
        # Sample and evaluate the posterior.
        data = benchmark.sample()
        log_posterior = benchmark.evaluate_log_joint(data['x'], lin)
        posterior = np.exp(log_posterior)

        # Evaluate the desired level.
        level = summaries.evaluate_credible_level(posterior, alpha)
        # Find the level that's close to the theta of interest.
        delta2 = np.square(data['theta'] - lin)
        assert delta2.shape == posterior.shape
        level0 = posterior.ravel()[np.argmin(delta2.ravel())]
        levels.append((level, level0))

    # Evaluate the number of successes and failures.
    level, level0 = np.transpose(levels)
    successes = np.sum(level > level0)
    pvalue = stats.binom_test(successes, len(level), alpha)
    assert pvalue > 0.01


def test_benchmark_stan_model():
    # Generate some data and fit the model.
    data = benchmark.sample()
    xs = data['x']
    model = benchmark.StanBenchmarkAlgorithm('summaries/benchmark.stan')
    samples, info = model.sample(np.asarray([xs.numpy()] * 3), 1000, keep_fits=True)

    # Validate the output and ensure the likelihood is the same in python and stan.
    assert samples.shape == (3, 1000, 1)
    fit = info['fits'][0]
    variables = fit.stan_variables()
    likelihood = benchmark.evaluate_gaussian_mixture_distribution(th.as_tensor(variables['theta']))
    # Make sure the parameters of the likelihood are the same.
    component_dist: th.distributions.Normal = likelihood._component_distribution
    np.testing.assert_allclose(variables['loc'], component_dist.loc[:, 0].abs())
    np.testing.assert_allclose(variables['scale'], component_dist.scale[:, 0])
    # Compare the likelihoods for each value.
    for x, part in zip(xs, variables['target_parts'].T):
        log_prob = likelihood.log_prob(x)
        np.testing.assert_allclose(part, log_prob, rtol=1e-5)


def test_benchmark_batch_sample():
    batch_size = 47
    batch = benchmark.sample(size=batch_size)
    assert batch['theta'].shape == (batch_size,)
    assert batch['x'].shape == (batch_size, benchmark.NUM_OBSERVATIONS)
    assert batch['noise'].shape == (batch_size, benchmark.NUM_NOISE_FEATURES)
