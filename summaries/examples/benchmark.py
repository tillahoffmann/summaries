import argparse
import matplotlib.figure
from matplotlib import pyplot as plt
import numpy as np
import pickle
from scipy import special
from tqdm import tqdm
import typing
from summaries.util import label_axes, trapznd


class NegativeBinomialDistribution:
    """
    Negative binomial distribution akin to `scipy.stats.nbinom` but faster.
    """
    def __init__(self, n, p):
        self.n = n
        self.p = p
        self._gammaln_n = special.gammaln(self.n)
        self._logp = np.log(self.p)
        self._log1mp = np.log1p(-self.p)

    def log_prob(self, x):
        return x * self._log1mp + self.n * self._logp + special.gammaln(x + self.n) \
            - self._gammaln_n - special.gammaln(x + 1)

    def sample(self, size=None):
        return np.random.negative_binomial(self.n, self.p, size)

    @property
    def mean(self):
        return self.n * (1 - self.p) / self.p


class UniformDistribution:
    """
    Uniform distribution.
    """
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def log_prob(self, x):
        return -np.log(self.upper - self.lower)

    def sample(self, size=None):
        return np.random.uniform(self.lower, self.upper, size)

    @property
    def mean(self):
        return (self.upper + self.lower) / 2


# Construct "nice" likelihoods given bivariate parameters in the sense that each likelihood is
# well-behaved over the unit box.
LIKELIHOODS = [
    lambda t1, t2: NegativeBinomialDistribution(1 + t1, 0.1 + 0.8 * t2),
    lambda t1, t2: NegativeBinomialDistribution(1 + t1, 0.1 + 0.8 * np.sqrt(t1 * t2)),
    lambda t1, t2: NegativeBinomialDistribution(1 + np.sqrt(t1 * t2), 0.1 + 0.8 * t1),
    lambda t1, t2: NegativeBinomialDistribution(1 / (1 + t1), 0.1 + 0.8 * t2),
    lambda *_: UniformDistribution(0, 1),
    lambda *_: UniformDistribution(0, 1),
]


def sample(likelihoods: list[typing.Callable], theta: np.ndarray, size: tuple = None) -> np.ndarray:
    """
    Draw samples with a given size from each likelihood for fixed parameters.

    Args:
        likelihoods: Iterable of functions that return likelihoods implementing `sample`.
        theta: Parameter values at which to sample.
        size: Sample shape.

    Returns:
        samples: Samples of shape `(len(likelihoods), *size)`.
    """
    return np.asarray([
        likelihood(*theta).sample(size) for likelihood in likelihoods
    ])


def evaluate_log_posterior(likelihoods: list[typing.Callable], samples: np.ndarray,
                           thetas: list[np.ndarray], normalize: bool = True) -> np.ndarray:
    """
    Evaluate the posterior numerically.

    Args:
        likelihoods: Iterable of functions that return likelihoods implementing `log_prob`.
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
        cumulative = cumulative + likelihood(*(t[..., None] for t in tt)).log_prob(x).sum(axis=-1)

    # Subtract maximum for numerical stability and normalize if desired.
    cumulative -= cumulative.max()
    if not normalize:
        return cumulative  # pragma: no cover
    norm = trapznd(np.exp(cumulative), *thetas)
    return cumulative - np.log(norm)


def _plot_example(likelihoods: list = None, n: int = 10, theta: np.ndarray = None) \
        -> matplotlib.figure.Figure:
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

    for i, (likelihood_fun, x) in enumerate(zip(likelihoods, xs)):
        color = f'C{i}'
        # Add a trailing dimension so we can evaluate the log probability in a batch.
        likelihood = likelihood_fun(*(t[..., None] for t in tt))
        if isinstance(likelihood, UniformDistribution):
            continue
        mean = likelihood.mean.squeeze()
        ax1.contour(*tt, mean, levels=[x.mean()], colors=color)

        # Show the posterior for each likelihood individually.
        log_posterior = evaluate_log_posterior([likelihood_fun], [x], [lin1, lin2], False)
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


def __entrypoint__(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, help='seed for the random number generator')
    parser.add_argument('num_samples', type=int, help='number of samples to generate')
    parser.add_argument('output', help='output file path')
    args = parser.parse_args(args)

    if args.seed is not None:
        np.random.seed(args.seed)

    # We dump each result into the same binary stream so we can load it again without loading the
    # entire file into memory (cf. https://stackoverflow.com/a/17623631/1150961).
    with open(args.output, 'wb') as fp:
        for _ in tqdm(range(args.num_samples)):
            theta = np.random.uniform(0, 1, 2)
            xs = sample(LIKELIHOODS, theta, 5)
            result = {
                'theta': theta,
                'xs': xs,
            }
            pickle.dump(result, fp)
