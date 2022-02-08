from matplotlib import pyplot as plt
import matplotlib.figure
import numpy as np
from scipy import stats


def _plot_example(a: float = 2, b: float = 5, n: int = 50, variance: float = 0.5, m: int = 10000) \
        -> matplotlib.figure.Figure:
    """
    Plot the prior, likelihood (posterior with improper prior), and posterior distribution for a
    simple conjugate model together with their entropies.

    Args:
        a: Shape parameter for the precision prior gamma distribution.
        b: Rate parameter for the precision prior gamma distribution.
        n: Number of observations in the dataset.
        variance: Realised variance in the dataset.
        m: Number of independent samples for evaluating the expected posterior entropy.
    """
    dists = {
        'Prior': stats.gamma(a, scale=1 / b),
        'Posterior with improper prior\n(normalized likelihood)':
            stats.gamma(n / 2, scale=2 / (n * variance)),
        'Posterior': stats.gamma(a + n / 2, scale=1 / (b + n * variance / 2)),
    }

    fig, ax = plt.subplots()
    eps = 1e-4
    for key, dist in dists.items():
        lin = np.linspace(*dist.ppf([eps, 1 - eps]))
        ax.plot(lin, dist.pdf(lin), label=f'{key} ($H={dist.entropy():.2f}$)')

    # Obtain samples from the generative model.
    precisions = dists['Prior'].rvs([m])
    xs = stats.norm(0, 1 / np.sqrt(precisions[:, None])).rvs([m, n])
    variances = xs.var(axis=1)
    posteriors = stats.gamma(a + n / 2, scale=1 / (b + n * variances / 2))

    # Show summary information.
    entropies = - posteriors.logpdf(precisions)
    mean = entropies.mean()
    std = entropies.std() / np.sqrt(m - 1)
    ax.plot([], [], color='none', label=fr'Expected entropy $H={mean:.2f}\pm{std:.2f}$')
    ax.legend()
    ax.set_xlabel(r'Precision $\tau$')
    ax.set_ylabel('Density')
    fig.tight_layout()
    return fig
