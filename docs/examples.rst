Examples
========

High posterior entropy compared with prior
------------------------------------------

The posterior entropy given a *particular* dataset may have higher entropy than the prior. But the
*expected* posterior entropy will always be at least as small as the prior entropy. Here, we
consider the simple example

.. math::

    \tau&\sim\mathrm{Gamma}(a, b)\\
    x\mid \tau&\sim\mathrm{Normal}(0, \tau^{-1})

which is a classic textbook example because the posterior is analytically tractable (see
:mod:`summaries.examples.bimodal` for details).

When the likelihood conflicts with the prior (ironically a situation where we acquire a lot of
information--and the KL divergence between prior and posterior is large), the posterior entropy may
be higher than the prior entropy as illustrated below.

.. plot:: summaries/examples/broad_posterior.py _plot_example


Bimodal posterior due to reparameterization
-------------------------------------------

In this example, we consider a simple conjugate model illustrating that the posterior mean is
difficult to interpret when the posterior is multimodal (e.g. mixture models) or more generally if
the posterior has symmetries. In particular, let

.. math::

    \tau \mid a, b &\sim \mathrm{Gamma}(a, b)\\
    x_i \mid \tau &\sim \mathrm{Normal}\left(0, \tau^{-1}\right),

where :math:`i` indexes each of the :math:`n` observations. This problem can be solved exactly
because the gamma prior for the precision (inverse variance) :math:`\tau` is conjugate to the normal
likelihood with known mean (zero in our case). In particular, the posterior is

.. math::

    \tau \mid a,b,x \sim\mathrm{Gamma}\left(a+\frac{n}{2}, b+\frac{1}{2} \sum_{i=1}^n x_i^2\right).

Rather than sticking with the standard parameterization, we instead change variables to
:math:`\theta` such that :math:`\tau=\theta^2` which induces a bimodal posterior with zero mean.

.. plot:: summaries/examples/bimodal.py _plot_example

Interface
^^^^^^^^^

.. automodule:: summaries.examples.bimodal
    :members:
