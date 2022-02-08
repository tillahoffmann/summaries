import matplotlib.figure
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
import typing
from summaries.util import label_axes, trapznd


# Construct "nice" likelihoods given bivariate parameters in the sense that each likelihood is
# well-behaved over the unit box.
LIKELIHOODS = [
    lambda t1, t2: stats.nbinom(1 + t1, 0.1 + 0.8 * t2),
    lambda t1, t2: stats.nbinom(1 + t1, 0.1 + 0.8 * np.sqrt(t1 * t2)),
    lambda t1, t2: stats.nbinom(1 + np.sqrt(t1 * t2), 0.1 + 0.8 * t1),
    lambda t1, t2: stats.nbinom(1 / (1 + t1), 0.1 + 0.8 * t2),
]


def sample(likelihoods: list[typing.Callable], theta: np.ndarray, size: tuple = None) -> np.ndarray:
    """
    Draw samples with a given size from each likelihood for fixed parameters.

    Args:
        likelihoods: Iterable of functions that return likelihoods implementing `rvs`.
        theta: Parameter values at which to sample.
        size: Sample shape.

    Returns:
        samples: Samples of shape `(len(likelihoods), *size)`.
    """
    return np.asarray([
        likelihood(*theta).rvs(size) for likelihood in likelihoods
    ])


def evaluate_log_posterior(likelihoods: list[typing.Callable], samples: np.ndarray,
                           thetas: list[np.ndarray], normalize: bool = True) -> np.ndarray:
    """
    Evaluate the posterior numerically.

    Args:
        likelihoods: Iterable of functions that return likelihoods implementing `logpmf`.
        samples: Samples at which to evaluate the likelihood.
        thetas: Locations at which to evaluate the posterior (assumed to cover the whole parameter
            space).
        normalize: Whether to normalize the likelihood.
    """
    # Evaluate the cumulative log likelihood.
    assert len(likelihoods) == len(samples)
    cumulative = 0
    tt = np.meshgrid(*thetas)
    for likelihood, x in zip(likelihoods, samples):
        cumulative = cumulative + likelihood(*(t[..., None] for t in tt)).logpmf(x).sum(axis=-1)

    # Subtract maximum for numerical stability and normalize if desired.
    cumulative -= cumulative.max()
    if not normalize:
        return cumulative  # pragma: no cover
    norm = trapznd(np.exp(cumulative), *thetas)
    return cumulative - np.log(norm)


def _plot_example(likelihoods: list = None, n: int = 10, theta: np.ndarray = None) \
        -> matplotlib.figure.Figure:  # pragma: no cover
    """
    Show (a) mean contours of different negative binomial distributions with level set by the mean
    of a sample drawn from the distribution and (b) individual posterior contours together with the
    posterior contour given all likelihoods.
    """
    # Validate arguments and draw a sample.
    likelihoods = likelihoods or LIKELIHOODS
    theta = (0.2, 0.5) if theta is None else theta
    xs = sample(likelihoods, theta, n)
    assert xs.shape == (len(likelihoods), n)

    # We use a different number of elements to raise a ValueError if we get the axes wrong.
    lin1 = np.linspace(0, 1, 100)
    lin2 = np.linspace(0, 1, 100)
    tt = np.asarray(np.meshgrid(lin1, lin2))

    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    ax1, ax2 = axes

    for i, (likelihood, x) in enumerate(zip(likelihoods, xs)):
        color = f'C{i}'
        # Add a trailing dimension so we can evaluate the logpmf in a batch.
        mean = likelihood(*(t[..., None] for t in tt)).mean().squeeze()
        ax1.contour(*tt, mean, levels=[x.mean()], colors=color)

        # Show the posterior for each likelihood individually.
        log_posterior = evaluate_log_posterior([likelihood], [x], [lin1, lin2], False)
        ax2.contourf(*tt, log_posterior, alpha=.15, colors=color, levels=[-2, 0])

    # And for the full posterior.
    log_posterior = evaluate_log_posterior(likelihoods, xs, [lin1, lin2], False)
    ax2.contour(*tt, log_posterior, colors='k', levels=[-2, -1], linestyles=['--', '-'])

    for ax in axes:
        ax.scatter(*theta, color='k', marker='x')
        ax.set_aspect('equal')
        ax.set_xlabel(r'Parameter $\theta_1$')
    ax1.set_ylabel(r'Parameter $\theta_2$')
    label_axes(axes, loc='top right')
    fig.tight_layout()
    return fig
