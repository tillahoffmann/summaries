import cmdstanpy
import matplotlib.figure
import matplotlib.patches
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import typing
from .algorithm import Algorithm
from .util import label_axes
from .distributions import Distribution, NormalDistribution, UniformDistribution


VARIANCE_OFFSET = 1.0
NUM_NOISE_FEATURES = 2


class GaussianMixtureDistribution(Distribution):
    """
    Mixture distribution of two Gaussians parameterized such that the data have zero mean and
    constant variance independent of parameters. But the likelihood is informative of the
    parameters.
    """
    def __init__(self, theta: np.ndarray) -> None:
        self.theta = theta
        # Construct the component distribution.
        loc = np.tanh(self.theta)
        scale = np.sqrt(VARIANCE_OFFSET - loc ** 2)
        self.component_dist = NormalDistribution(loc, scale)

    def sample(self, size: tuple = None) -> np.ndarray:
        # Sample from the component distribution, then randomly assign to the +ve or -ve component.
        x = self.component_dist.sample(size)
        return x * np.random.choice([-1, 1], x.shape)

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        log_probs = [
            self.component_dist.log_prob(x),
            self.component_dist.log_prob(-x),
        ]
        return np.logaddexp(*log_probs) - np.log(2)


LIKELIHOODS = {
    'gaussian_mixture': GaussianMixtureDistribution,
    'noise': lambda theta: UniformDistribution(
        np.zeros(np.shape(theta) + (NUM_NOISE_FEATURES,)),
        np.ones(np.shape(theta) + (NUM_NOISE_FEATURES,))
    ),
}


def sample(likelihoods: list[typing.Callable], theta: np.ndarray, size: tuple = None) -> np.ndarray:
    """
    Draw samples with a given size from each likelihood for fixed parameters.

    Args:
        likelihoods: Mapping of functions that return likelihoods implementing `sample`.
        theta: Parameter values at which to sample.
        size: Sample shape.

    Returns:
        samples: Mapping of samples of shape `(*size, p)` keyed by likelihood name, where `p` is the
            sum of the dimensionality of each likelihood.
    """
    return {key: value(theta).sample(size) for key, value in likelihoods.items()}


def evaluate_log_joint(likelihoods: list[typing.Callable], samples: np.ndarray, theta: np.ndarray,
                       normalize: bool = True) -> np.ndarray:
    """
    Evaluate the log joint distribution numerically over a grid.

    Args:
        likelihoods: Iterable of functions that return likelihoods implementing `log_prob`.
        samples: Samples at which to evaluate the log joint.
        theta: Parameter values at which to evaluate the log joint (assumed to cover the whole
            parameter comain).
        normalize: Whether to normalize the log joint with respect to the parameters, yielding a
            posterior.
    """
    missing = set(samples) - set(likelihoods)
    assert not missing, f'there is no likelihood for samples: {", ".join(missing)}'

    # Evaluate the prior.
    log_joint = NormalDistribution(0, 1).log_prob(theta)
    for key, x in samples.items():
        # Evaluate the log probability for this likelihood and aggregate the trailing dimensions.
        likelihood = likelihoods[key](theta[..., None])
        log_prob = likelihood.log_prob(x)
        axis = tuple(np.arange(theta.ndim, log_prob.ndim))
        log_joint = log_joint + log_prob.sum(axis=axis)

    # Sanity check that the log joint has the right shape.
    assert log_joint.shape == theta.shape

    # Subtract maximum for numerical stability and normalize if desired.
    log_joint -= log_joint.max()
    if not normalize:
        return log_joint  # pragma: no cover
    norm = np.trapz(np.exp(log_joint), theta)
    return log_joint - np.log(norm)


def _plot_example(likelihoods: list = None, n: int = 10, theta: np.ndarray = 1.5) \
        -> matplotlib.figure.Figure:
    """
    Show (a) mixture likelihood with rug plot of data and (b) posterior dsitribution.
    """
    # Validate arguments and draw a sample.
    likelihoods = likelihoods or LIKELIHOODS
    data = sample(likelihoods, theta, n)
    xs = data['gaussian_mixture']
    lin = np.linspace(-3, 3, 100)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Show the data and likelihood.
    ax1.scatter(xs, np.zeros_like(xs), marker='|', color='k')
    likelihood: GaussianMixtureDistribution = likelihoods['gaussian_mixture'](theta)
    ax1.plot(lin, np.exp(likelihood.log_prob(lin)))
    ax1.set_xlabel('Data $x$')
    ax1.set_ylabel(r'Likelihood $p(x\mid\theta)$')

    # Show the posterior.
    log_posterior = evaluate_log_joint(likelihoods, data, lin)
    ax2.plot(lin, np.exp(log_posterior))
    ax2.axvline(theta, color='k', ls=':')
    ax2.set_ylabel(r'Posterior $p(\theta\mid x)$')
    ax2.set_xlabel(r'Parameter $\theta$')

    label_axes([ax1, ax2])
    fig.tight_layout()
    return fig


class StanBenchmarkAlgorithm(Algorithm):
    """
    Stan implementation of the benchmark model.
    """
    def __init__(self, path=None):
        self.path = path or __file__.replace('.py', '.stan')
        self.model = cmdstanpy.CmdStanModel(stan_file=self.path)

    def sample(self, data: np.ndarray, num_samples: int, show_progress: bool = True,
               keep_fits: bool = False, **kwargs) -> typing.Tuple[np.ndarray, dict]:
        # Set default arguments.
        kwargs = {
            'chains': 1,
            'iter_sampling': num_samples,
            'show_progress': False,
            'sig_figs': 9,
        } | kwargs
        samples = []
        info = {}
        for x in tqdm(data) if show_progress else data:
            # Fit the model.
            x: np.ndarray
            stan_data = {
                'num_obs': x.shape[0],
                'x': x,
                'variance_offset': VARIANCE_OFFSET,
            }
            fit = self.model.sample(stan_data, **kwargs)

            # Extract the samples and store the fits.
            samples.append(fit.stan_variable('theta'))
            if keep_fits:
                info.setdefault('fits', []).append(fit)

        # We append a trailing dimension of one element for consistency with the other algorithms.
        return np.asarray(samples)[..., None], info

    @property
    def num_params(self):
        return 1
