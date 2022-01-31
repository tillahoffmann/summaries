from matplotlib import pyplot as plt
import numpy as np
from scipy import stats


def _plot_example(a: float = 10, b: float = 50, n: int = 100, variance: float = 0.5) \
        -> None:  # pragma: no cover
    """
    Plot the prior, likelihood (posterior with improper prior), and posterior distribution for a
    simple conjugate model together with their entropies.

    Args:
        a: Shape parameter for the precision prior gamma distribution.
        b: Rate parameter for the precision prior gamma distribution.
        n: Number of observations in the dataset.
        variance: Realised variance in the dataset.
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
    m = 10000
    precisions = dists['Prior'].rvs([m])
    xs = stats.norm(0, 1 / np.sqrt(precisions[:, None])).rvs([m, n])
    variances = xs.var(axis=1)
    posteriors = stats.gamma(a + n / 2, scale=1 / (b + n * variances / 2))

    # Show summary information.
    mean = posteriors.entropy().mean()
    std = posteriors.entropy().std() / np.sqrt(m - 1)
    ax.plot([], [], color='none', label=fr'Expected entropy $H={mean:.3f}\pm{std:.3f}$')
    ax.legend()
    ax.set_xlabel(r'Precision $\tau$')
    ax.set_ylabel('Density')
    fig.tight_layout()
