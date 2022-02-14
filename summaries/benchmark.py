import cmdstanpy
import matplotlib.figure
import matplotlib.patches
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import typing
from .algorithm import Algorithm
from .util import label_axes, trapznd, softplus
from .distributions import Distribution, MultiNormalDistribution, UniformDistribution


class CorrelatedGaussianMixtureDistribution(Distribution):
    """
    Mixture distribution of two correlated multivariate Gaussians such that the data have constant
    mean, variance, and covariance independent of parameters. But the likelihood is informative of
    the parameters.
    """
    def __init__(self, t1: np.ndarray, t2: np.ndarray, epsilon: float = 1e-6) -> None:
        self.t1 = np.asarray(t1)
        self.t2 = np.asarray(t2)
        self.radius = np.sqrt(self.t1 ** 2 + self.t2 ** 2)
        self.loc = 2 + self.radius
        self.scale = np.sqrt(softplus(25 - self.loc ** 2))
        self.corr = self.t1 / self.radius

        # Construct location and covariances for the "positive" and "negative" componentns of the
        # mixture.
        self._loc_plus = np.ones(2) * self.loc[..., None]
        self._loc_minus = -np.ones(2) * self.loc[..., None]
        self._cov_plus = np.ones((2, 2)) * self.scale[..., None, None] ** 2
        self._cov_plus[..., 0, 1] *= self.corr
        self._cov_plus[..., 1, 0] *= self.corr
        self._cov_plus[..., 0, 0] += epsilon
        self._cov_plus[..., 1, 1] += epsilon
        self._cov_minus = self._cov_plus.copy()
        self._cov_minus[..., 0, 1] *= -1
        self._cov_minus[..., 1, 0] *= -1
        self._dist_plus = MultiNormalDistribution(self._loc_plus, self._cov_plus)
        self._dist_minus = MultiNormalDistribution(self._loc_minus, self._cov_minus)

    def sample(self, size: tuple = None) -> np.ndarray:
        x_plus = self._dist_plus.sample(size)
        x_minus = self._dist_minus.sample(size)
        return np.where(np.random.choice(2, x_plus.shape[:-1])[..., None], x_plus, x_minus)

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        log_prob_plus = self._dist_plus.log_prob(x)
        log_prob_minus = self._dist_minus.log_prob(x)
        return np.logaddexp(log_prob_plus, log_prob_minus) - np.log(2)


LIKELIHOODS = {
    'gaussian_mixture': CorrelatedGaussianMixtureDistribution,
    'noise': lambda t1, _: UniformDistribution(
        np.zeros(np.shape(t1) + (2,)),
        np.ones(np.shape(t1) + (2,))
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
        samples: Samples of shape `(*size, p)`, where `p` is the sum of the dimensionality of each
            likelihood.
    """
    return {key: value(*theta).sample(size) for key, value in likelihoods.items()}


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
    missing = set(samples) - set(likelihoods)
    assert not missing, f'there is no likelihood for samples: {", ".join(missing)}'
    tt = np.meshgrid(*thetas)
    cumulative = - (tt[0] ** 2 + tt[1] ** 2) / 2
    for key, x in samples.items():
        likelihood = likelihoods[key](*(t[..., None] for t in tt))
        log_prob = likelihood.log_prob(x)
        # Sum over everything that isn't the dimensions of the parameters.
        axis = tuple(np.arange(len(thetas), log_prob.ndim))
        cumulative = cumulative + log_prob.sum(axis=axis)

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
    theta = (1, 0.5) if theta is None else theta
    xs = sample(likelihoods, theta, n)

    # We use a different number of elements to raise a ValueError if we get the axes wrong.
    lin1 = np.linspace(-2, 2, 100)
    lin2 = np.linspace(-2, 2, 101)
    tt = np.asarray(np.meshgrid(lin1, lin2))

    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Show the data.
    ax1.scatter(*xs['gaussian_mixture'].T, marker='.')
    ax1.set_aspect('equal')
    label_axes(ax1, '(a)')
    ax1.set_xlabel('Data $x_1$')
    ax1.set_ylabel('Data $x_2$')

    # Show the likelihood.
    likelihood: CorrelatedGaussianMixtureDistribution = likelihoods['gaussian_mixture'](*theta)
    args = [
        (likelihood._loc_minus, likelihood._cov_minus),
        (likelihood._loc_plus, likelihood._cov_plus),
    ]
    for loc, cov in args:
        evals, evecs = np.linalg.eigh(cov)
        angle = np.arctan2(*evecs[:, 0])  # This may not quite be correct but does the trick.
        ellipse = matplotlib.patches.Ellipse(loc, *2.5 * np.sqrt(evals), facecolor='none',
                                             edgecolor='k', ls='--', angle=np.rad2deg(angle))
        ax1.add_patch(ellipse)

    # Show the posterior distribution.
    log_posterior = evaluate_log_posterior(likelihoods, xs, [lin1, lin2], False)
    ax2.pcolormesh(*tt, np.exp(log_posterior))
    ax2.scatter(*theta, color='k', marker='X').set_edgecolor('w')
    ax2.set_ylabel(r'Parameter $\theta_2$')
    ax2.set_xlabel(r'Parameter $\theta_1$')

    ax2.set_aspect('equal')
    label_axes(ax2, '(b)', color='w')
    fig.tight_layout()
    return fig


class StanBenchmarkAlgorithm(Algorithm):
    """
    Stan implementation of the benchmark model.
    """
    def __init__(self, path=None, epsilon: float = 1e-6):
        self.path = path or __file__.replace('.py', '.stan')
        self.model = cmdstanpy.CmdStanModel(stan_file=self.path)
        self.epsilon = epsilon

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
                'n': x.shape[0],
                'x': x,
                'epsilon': self.epsilon,
            }
            fit = self.model.sample(stan_data, **kwargs)

            # Extract the samples and store the fits.
            samples.append(fit.stan_variable('theta'))
            if keep_fits:
                info.setdefault('fits', []).append(fit)

        return np.asarray(samples), info

    @property
    def num_params(self):
        return 2
