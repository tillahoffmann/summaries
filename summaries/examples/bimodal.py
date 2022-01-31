from matplotlib import pyplot as plt
import numpy as np
from scipy import special


def evaluate_log_prob(theta: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    r"""
    Evaluate the log probability of the :math:`\theta` parameter such that :math:`\theta^2` follows
    a gamma distribution with shape :math:`a` and rate :math:`b`.

    Args:
        theta: Parameter value.
        a: Shape of the gamma distribution.
        b: Rate of the gamma distribution.
    """
    theta = np.abs(theta)
    return a * np.log(b) - special.gammaln(a) + (2 * a - 1) * np.log(theta) - b * theta ** 2


def evaluate_entropy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    r"""
    Evaluate the posterior entropy of the parameter :math:`\theta` such that :math:`\theta^2`
    follows a gamma distribution with shape :math:`a` and rate :math:`b`.

    Args:
        a: Shape of the gamma distribution.
        b: Rate of the gamma distribution.
    """
    return a - np.log(b) / 2 + special.gammaln(a) + (1 / 2 - a) * special.digamma(a)


def evaluate_expected_posterior_entropy(a: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:
    r"""
    Evaluate the expected posterior entropy of the parameter :math:`\theta` such that
    :math:`\tau=\theta^2` follows a gamma distribution with shape :math:`a` and rate :math:`b` a
    priori. Then :math:`n` samples :math:`x` are drawn from a zero-mean normal distribution with
    precision :math:`\tau`, and we seek to infer :math:`\theta`.

    Args:
        a: Shape of the gamma distribution.
        b: Rate of the gamma distribution.
        n: Number of observations per experiment.
    """
    return (2 * a + n - np.log(b) + 2 * special.gammaln(a + n / 2) + special.digamma(a)
            - (2 * a + n) * special.digamma(a + n / 2)) / 2


def _plot_example() -> None:  # pragma: no cover
    r"""
    Plot the posterior distribution of :math:`\theta` given synthetic data.

    Args:
        a: Shape parameter for the precision prior gamma distribution.
        b: Rate parameter for the precision prior gamma distribution.
        n: Number of observations in the dataset.
    """
    np.random.seed(0)

    # Generate data.
    a, b, n = 3, 4, 10
    tau = np.random.gamma(a, 1 / b)
    x = np.random.normal(0, 1 / np.sqrt(tau), n)

    # Infer the posterior and show it.
    ap = a + n / 2
    bp = b + np.sum(x ** 2)
    lin = np.linspace(-1.5, 1.5, 100)

    fig, ax = plt.subplots()
    ax.plot(lin, np.exp(evaluate_log_prob(lin, ap, bp)))
    ax.set_xlabel(r'Parameter $\theta$')
    ax.set_ylabel(r'Posterior density $p(\theta\mid a,b,x)$')
    fig.tight_layout()
