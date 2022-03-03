Examples
========

High posterior entropy compared with prior
------------------------------------------

The posterior entropy given a *particular* dataset may have higher entropy than the prior. But the *expected* posterior entropy will always be at least as small as the prior entropy. Here, we consider the simple example

.. math::

    \tau&\sim\mathrm{Gamma}(a, b)\\
    x\mid \tau&\sim\mathrm{Normal}(0, \tau^{-1})

which is a classic textbook example because the posterior is analytically tractable (see :ref:`bimodal` for details).

When the likelihood conflicts with the prior (ironically a situation where we acquire a lot of information--and the KL divergence between prior and posterior is large), the posterior entropy may be higher than the prior entropy as illustrated below.

.. plot::

    from summaries.examples.broad_posterior import _plot_example
    _plot_example()

Methods that minimise the posterior given the dataset, such as proposed by Nunes and Balding (2010), will thus not select *any* feature for inference.

.. _bimodal:

Bimodal posterior due to reparameterization
-------------------------------------------

In this example, we consider a simple conjugate model illustrating that the posterior mean is difficult to interpret when the posterior is multimodal (e.g. mixture models) or more generally if the posterior has symmetries. In particular, let

.. math::

    \tau \mid a, b &\sim \mathrm{Gamma}(a, b)\\
    x_i \mid \tau &\sim \mathrm{Normal}\left(0, \tau^{-1}\right),

where :math:`i` indexes each of the :math:`n` observations. This problem can be solved exactly because the gamma prior for the precision (inverse variance) :math:`\tau` is conjugate to the normal likelihood with known mean (zero in our case). In particular, the posterior is

.. math::

    \tau \mid a,b,x \sim\mathrm{Gamma}\left(a+\frac{n}{2}, b+\frac{1}{2} \sum_{i=1}^n x_i^2\right).

Rather than sticking with the standard parameterization, we instead change variables to :math:`\theta` such that :math:`\tau=\theta^2` which induces a bimodal posterior with zero mean.

.. plot::

    from summaries.examples.bimodal import _plot_example
    _plot_example()

Importance of prior information for mutual information
------------------------------------------------------

Chen et al. (2021) suggest that the features selected by maximising the mutual informationb between the model parameters :math:`\theta` and summary statistics :math:`t` does not depend on the prior. This proposition is not true as can be seen from a simple example. Suppose the piecewise likelihood (see bottom left panel)

.. math::

    x\sim\begin{cases}
        \mathrm{Normal}\left(0, \exp\left(\frac{\theta}{2}\right)\right) &\text{if }\theta\leq 0\\
        \mathrm{Normal}\left(\theta, 1\right)&\text{if }\theta>0
    \end{cases}

together with three different priors :math:`\theta\sim\mathrm{Normal}(\{-1,0,1\}, \sigma^2)`--one to the left of the transition at :math:`\theta=0`, one at the transition, and one to the right of the transition (see top left panel).

We consider two summary statistics, the sample mean :math:`\bar x` and the natural logarithm of the sample variance :math:`\log \mathrm{var} x`. Intuitively, the former will be informative when :math:`\theta > 0` and the latter will be informative when :math:`\theta<0`. To thest this hypothesis, we evaluate the mutual information between the parameter and the two summary statistics, for each of the priors. Scatter plot of the summary statistics against parameter values are shown in the right column together with mutual information estimates. Indeed, the "best" summary statistic is prior dependent as :math:`\bar x` is not informative for the left prior and :math:`\log \mathrm{var} x` is not informative for the right prior. The central prior requires both summary statistics to infer the parameter.

.. plot::

    from summaries.examples.piecewise_likelihood import _plot_example
    _plot_example()

Benchmark problem
-----------------

We need a ground truth to compare with if we want to evaluate different methods for extracting useful summary statistics. We consider a model with one parameter :math:`\theta` drawn from a standard normal distribution. The likelihood is a mixture of normal distributions, as shown in panel (a), constructed such that the first four moments only contain minimal information about the parameters. Sufficient statistics do not exist because the mixture distribution does not belong to the exponential family. Specifically,

    .. math::

        x &\sim \frac{1}{2}\left(\mathrm{Normal}(\tanh(\theta),\sigma^2) + \mathrm{Normal}(-\tanh(\theta),\sigma^2)\right),\\
        \text{where } \sigma^2 &= 1 - \tanh^2(\theta).

The posterior is bimodal because of the symmetry in the likelihood, as shown in panel (b).

.. plot::

    from summaries.benchmark import _plot_example
    import numpy as np
    np.random.seed(0)
    _plot_example()
