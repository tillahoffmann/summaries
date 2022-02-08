import numpy as np
from pytest_bootstrap import bootstrap_test
from scipy import integrate, stats
import summaries
from summaries.examples import benchmark, bimodal, broad_posterior, piecewise_likelihood  \
    # noqa: F401


def test_expected_posterior_entropy():
    m = 10000  # Number of independent samples for bootstrapping.
    n = 10  # Number of observations per experiment.
    a = 3  # Prior shape parameter.
    b = 4  # Prior rate parameter.

    # Generate `m` independent realisations of the process.
    tau = np.random.gamma(a, 1 / b, m)
    xs = np.random.normal(0, 1 / np.sqrt(tau[:, None]), (m, n))

    # Perform inference and evaluate the entropies.
    ap = a + n / 2
    bp = b + np.sum(xs ** 2, axis=1) / 2
    entropies = bimodal.evaluate_entropy(ap, bp)

    # Compare with theory using a bootstrap test.
    bootstrap_test(entropies, np.mean, bimodal.evaluate_expected_posterior_entropy(a, b, n))


def test_posterior_entropy():
    a = 3
    b = 4

    def target(x):
        log_prob = bimodal.evaluate_log_prob(x, a, b)
        return -log_prob * np.exp(log_prob)

    actual = bimodal.evaluate_entropy(a, b)
    desired, _ = integrate.quad(target, -np.inf, np.inf)
    np.testing.assert_allclose(actual, desired)


def test_log_prob_norm():
    a = 3
    b = 4
    actual, _ = integrate.quad(lambda x: np.exp(bimodal.evaluate_log_prob(x, a, b)), -np.inf,
                               np.inf)
    np.testing.assert_allclose(actual, 1)


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
