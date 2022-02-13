import numpy as np
from scipy import stats
import summaries
from summaries import benchmark


def test_benchmark_coverage():
    """
    This test verifies that the exact posterior (based on numerical integration) has correct
    coverage in the sense that the `1 - alpha` credible interval contains the true value in
    `1 - alpha` of the simulated cases.
    """
    alpha = .3
    thetas = [np.linspace(-4, 4, n) for n in [100, 101]]
    levels = []
    for _ in range(100):
        # Sample and evaluate the posterior.
        theta = np.random.normal(0, 1, 2)
        xs = benchmark.sample(benchmark.LIKELIHOODS, theta, 10)
        log_posterior = benchmark.evaluate_log_posterior(benchmark.LIKELIHOODS, xs, thetas)
        posterior = np.exp(log_posterior)

        # Evaluate the desired level.
        level = summaries.evaluate_credible_level(posterior, alpha)
        # Find the level that's close to the theta of interest.
        delta2 = np.square(theta[0] - thetas[0]) + np.square(theta[1] - thetas[1])[:, None]
        assert delta2.shape == posterior.shape
        level0 = posterior.ravel()[np.argmin(delta2.ravel())]
        levels.append((level, level0))

    # Evaluate the number of successes and failures.
    level, level0 = np.transpose(levels)
    successes = np.sum(level > level0)
    pvalue = stats.binom_test(successes, len(level), alpha)
    assert pvalue > 0.01


def test_benchmark_stan_model():
    # Generate some data.
    theta = np.random.uniform(0, 1, 2)
    num_samples = 7
    sample = benchmark.sample(benchmark.LIKELIHOODS, theta, num_samples)
    xs = sample['gaussian_mixture']
    model = benchmark.StanBenchmarkAlgorithm('summaries/benchmark.stan')
    samples, info = model.sample(np.asarray([xs, xs, xs]), 1000, keep_fits=True)

    # Validate the output and ensure the likelihood is the same in python and stan.
    assert samples.shape == (3, 1000, 2)
    fit = info['fits'][0]
    variables = fit.stan_variables()
    likelihood = benchmark.LIKELIHOODS['gaussian_mixture'](*variables['theta'].T)
    # Make sure the parameters of the likelihood are the same.
    np.testing.assert_allclose(variables['covp'], likelihood._cov_plus)
    np.testing.assert_allclose(variables['covm'], likelihood._cov_minus)
    np.testing.assert_allclose(variables['locp'], likelihood._loc_plus)
    np.testing.assert_allclose(variables['locm'], likelihood._loc_minus)
    # Compare the likelihoods for each value.
    for x, part in zip(xs, variables['parts'].T):
        log_prob = likelihood.log_prob(x)
        np.testing.assert_allclose(part, log_prob)
