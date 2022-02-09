import matplotlib.figure
import numpy as np
from scipy import stats
import summaries
from summaries import benchmark
from unittest import mock
from summaries.scripts import generate_benchmark_data


def test_benchmark_plot():
    assert isinstance(benchmark._plot_example(), matplotlib.figure.Figure)


def test_benchmark_coverage():
    """
    This test verifies that the exact posterior (based on numerical integration) has correct
    coverage in the sense that the `1 - alpha` credible interval contains the true value in
    `1 - alpha` of the simulated cases.
    """
    alpha = .3
    thetas = [np.linspace(0, 1, n) for n in [100, 101]]
    levels = []
    for _ in range(100):
        # Sample and evaluate the posterior.
        theta = np.random.uniform(0, 1, 2)
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


def test_negative_binomial_distribution():
    n = 10
    p = .3
    dist1 = stats.nbinom(n, p)
    dist2 = benchmark.NegativeBinomialDistribution(n, p)
    x = dist2.sample()
    np.testing.assert_allclose(dist1.logpmf(x), dist2.log_prob(x))
    np.testing.assert_allclose(dist1.mean(), dist2.mean)


def test_uniform_distribution():
    lower = 3
    upper = 5
    dist1 = stats.uniform(lower, upper - lower)
    dist2 = benchmark.UniformDistribution(lower, upper)
    x = dist2.sample()
    np.testing.assert_allclose(dist1.logpdf(x), dist2.log_prob(x))
    np.testing.assert_allclose(dist1.mean(), dist2.mean)


def test_generate_benchmark_data():
    with mock.patch('builtins.open') as open_, mock.patch('pickle.dump') as dump_:
        generate_benchmark_data.__main__(['--seed=0', '23', 'some_file.pkl'])

    open_.assert_called_once_with('some_file.pkl', 'wb')
    dump_.assert_called_once()
    (result, _), _ = dump_.call_args
    for key in ['xs', 'theta']:
        assert key in result
        assert len(result[key]) == 23
    assert 'args' in result
