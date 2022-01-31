r"""
High posterior entropy compared with prior
------------------------------------------

The posterior entropy given a *particular* dataset may have higher entropy than the prior. But the
*expected* posterior entropy will always be at least as small as the prior entropy. Here, we
consider a simple example

.. math::

    \tau&\sim\mathrm{Gamma}(a, b)\\
    x\mid \tau&\sim\mathrm{Normal}(0, \tau^{-1})

which is a classic textbook example because the posterior is analytically tractable (see
:mod:`summaries.examples.bimodal` for details).

When the likelihood conflicts with the prior (ironically a situation where we acquire a lot of
information--and the KL divergence between prior and posterior is large), the posterior entropy may
be higher than the prior entropy as illustrated below.

.. plot::
    :include-source:

    import numpy as np
    from scipy import stats

    a = 10
    b = 50
    variance = .5
    n = 100

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
"""

from matplotlib import axes
axes.Axes.legend
