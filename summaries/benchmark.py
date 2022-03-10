import cmdstanpy
import matplotlib.figure
import matplotlib.patches
from matplotlib import pyplot as plt
import numpy as np
import torch as th
from tqdm import tqdm
import typing
from .algorithm import Algorithm
from .nn import DenseCompressor, DenseStack
from .util import label_axes, normalize_shape


VARIANCE_OFFSET = 1.0
NUM_NOISE_FEATURES = 2
NUM_OBSERVATIONS = 10


def evaluate_gaussian_mixture_distribution(theta: th.Tensor) -> th.distributions.MixtureSameFamily:
    """
    Mixture distribution of two Gaussians parameterized such that the data have zero mean and
    constant variance independent of parameters. But the likelihood is informative of the
    parameters.
    """
    loc = theta.tanh()
    scale = (VARIANCE_OFFSET - loc.square()).sqrt()
    return th.distributions.MixtureSameFamily(
        th.distributions.Categorical(th.ones((*theta.shape, 2)) / 2),
        th.distributions.Normal((2 * th.arange(2) - 1) * loc[..., None], scale[..., None])
    )


def sample(*, theta: th.Tensor = None, size: tuple = None, num_observations: int = None,
           num_noise_features: int = None) -> th.Tensor:
    """
    Draw samples with a given size from each likelihood for fixed parameters.

    Args:
        theta: Parameter values at which to sample (drawn from the prior if not given).
        size: Sample shape.
        num_observations: Number of observations per sample.
        num_noise_features: Number of noise features per sample.

    Returns:
        samples: Mapping of samples of shape `(*size, p)` keyed by likelihood name, where `p` is the
            sum of the dimensionality of each likelihood.
    """
    size = normalize_shape(size)
    if theta is None:
        theta = th.distributions.Normal(0, 1).sample(size)
    num_noise_features = num_noise_features or NUM_NOISE_FEATURES
    num_observations = num_observations or NUM_OBSERVATIONS
    x = evaluate_gaussian_mixture_distribution(theta).sample((num_observations,))
    x = x.moveaxis(0, -1)

    noise_shape = (*size, num_observations, num_noise_features)
    return {
        # Add a trailing dimension of size one for consistency with multidimensional setups.
        'theta': theta[..., None],
        'x': x[..., None],
        'noise': th.distributions.Normal(0, 1).sample(noise_shape),
    }


def evaluate_log_joint(x: th.Tensor, theta: th.Tensor, normalize: bool = True) -> th.Tensor:
    """
    Evaluate the log joint distribution numerically over a grid.

    Args:
        x: Samples at which to evaluate the log joint.
        theta: Parameter values at which to evaluate the log joint (assumed to cover the whole
            parameter comain).
        normalize: Whether to normalize the log joint with respect to the parameters, yielding a
            posterior.
    """
    # Evaluate the prior.
    log_prior = th.distributions.Normal(0.0, 1.0).log_prob(theta)
    # Evaluate the likelihood. We expand the dimension so we can sum over the samples.
    dist = evaluate_gaussian_mixture_distribution(theta[..., None])
    log_likelihood = dist.log_prob(x[..., 0]).sum(axis=-1)
    log_joint = log_prior + log_likelihood

    # Subtract maximum for numerical stability and normalize if desired.
    log_joint -= log_joint.max()
    if not normalize:
        return log_joint  # pragma: no cover
    norm = th.trapz(log_joint.exp(), theta)
    return log_joint - norm.log()


def _plot_example(theta: th.Tensor = None) -> matplotlib.figure.Figure:
    """
    Show (a) mixture likelihood with rug plot of data and (b) posterior dsitribution.
    """
    # Validate arguments and draw a sample.
    if theta is None:
        theta = th.scalar_tensor(1.5)
    data = sample(theta=theta)
    lin = th.linspace(-3, 3, 100)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Show the data and likelihood.
    ax1.scatter(data['x'], np.zeros_like(data['x']), marker='|', color='k')
    likelihood = evaluate_gaussian_mixture_distribution(theta)
    ax1.plot(lin, likelihood.log_prob(lin).exp())
    ax1.set_xlabel('Data $x$')
    ax1.set_ylabel(r'Likelihood $p(x\mid\theta)$')

    # Show the posterior.
    log_posterior = evaluate_log_joint(data['x'], lin)
    ax2.plot(lin, log_posterior.exp())
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

    def sample(self, data: th.Tensor, num_samples: int, show_progress: bool = True,
               keep_fits: bool = False, **kwargs) -> typing.Tuple[th.Tensor, dict]:
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
            stan_data = {
                'num_obs': x.shape[0],
                'x': x[..., 0],
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


class MDNBenchmarkAlgorithm(th.nn.Module):
    """
    Simple Gaussian mixture density network for the benchmark problem.

    Args:
        num_components: Number of components of the Gaussian mixture model.
        num_features: Number of hidden features serving as summary statistics.
    """
    def __init__(self, num_components: int, num_features: int) -> None:
        super().__init__()
        self.num_features = num_features
        self.num_components = num_components
        self.compressor = DenseCompressor([1, 16, 16, num_features], th.nn.Tanh())
        self.logit_layers = DenseStack([num_features, 16, num_components], th.nn.Tanh())
        self.loc_layers = DenseStack([num_features, 16, num_components], th.nn.Tanh())
        self.log_scale_layers = DenseStack([num_features, 16, num_components], th.nn.Tanh())

    def forward(self, x: th.Tensor) -> th.distributions.MixtureSameFamily:
        # Compress the data.
        *batch_size, _num_obs, _num_dims = x.shape
        x = self.compressor(x)
        assert x.shape == (*batch_size, self.num_features)

        # Estimate properties of the Gaussian mixture.
        logits = self.logit_layers(x)
        locs = self.loc_layers(x)
        scales = self.log_scale_layers(x).exp()
        for x in logits, locs, scales:
            assert x.shape == (*batch_size, self.num_components)

        dist = th.distributions.MixtureSameFamily(
            th.distributions.Categorical(logits=logits),
            th.distributions.Normal(locs, scales),
        )
        return dist
