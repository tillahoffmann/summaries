r"""
Bimodal posterior due to reparameterization
-------------------------------------------

In this example, we consider a simple conjugate model illustrating that the posterior mean is
difficult to interpret when the posterior is multimodal (e.g. mixture models) or more generally if
the posterior has symmetries. In particular, let

.. math::

    \tau \mid a, b &\sim \mathrm{Gamma}(a, b)\\
    x_i \mid \theta &\sim \mathrm{Normal}\left(0, \tau^{-1}\right),

where :math:`i` indexes each of the :math:`n` observations. This problem can be solved exactly
because the gamma prior for the precision (inverse variance) :math:`\tau` is conjugate to the normal
likelihood with known mean (zero in our case). In particular, the posterior is

.. math::

    \tau \mid a,b,x \sim\mathrm{Gamma}\left(a+\frac{n}{2}, b+\frac{1}{2} \sum_{i=1}^n x_i^2\right).

Rather than sticking with the standard parameterization, we instead change variables to
:math:`\theta` such that :math:`\tau=\theta^2` which induces a bimodal posterior with zero mean.

.. plot::
   :include-source:

   import numpy as np
   from summaries.examples import bimodal
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
   ax.plot(lin, np.exp(bimodal.evaluate_log_prob(lin, ap, bp)))
   ax.set_xlabel(r'Parameter $\theta$')
   ax.set_ylabel(r'Posterior density $p(\theta\mid a,b,x)$')
   fig.tight_layout()
"""

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
