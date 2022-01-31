import numpy as np
from pytest_bootstrap import bootstrap_test
from scipy import integrate
from summaries.examples import bimodal, broad_posterior  # noqa: F401


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
