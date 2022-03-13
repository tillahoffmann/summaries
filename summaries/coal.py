import torch as th
from .nn import DenseStack


class MixtureDensityNetwork(th.nn.Module):
    """
    Simple Gaussian mixture density network for the benchmark problem.

    Args:
        compressor: Module to compress data to summary statistics.
        expansion_nodes: Number of nodes in hidden layers to expand from statistics to mixture
            density parameters. The first number of nodes must match the number of statistics. The
            last number of nodes is the number of components of the mixture.
    """
    def __init__(self, compressor: th.nn.Module, expansion_nodes: list[int],
                 activation: th.nn.Module) -> None:
        super().__init__()
        self.compressor = compressor
        # Build stacks for mixture weights and beta-distribution concentration parameters.
        self.logits = DenseStack(expansion_nodes, activation)
        self.log_a = DenseStack([*expansion_nodes[:-1], 2 * expansion_nodes[-1]], activation)
        self.log_b = DenseStack([*expansion_nodes[:-1], 2 * expansion_nodes[-1]], activation)

    def forward(self, x: th.Tensor) -> th.distributions.Distribution:
        # Compress the data and estimate properties of the Gaussian copula mixture.
        y: th.Tensor = self.compressor(x)
        logits = self.logits(y)
        a = self.log_a(y).exp()
        b = self.log_b(y).exp()
        a = a.reshape((*a.shape[:-1], -1, 2))
        b = b.reshape((*b.shape[:-1], -1, 2))

        component_distribution = th.distributions.TransformedDistribution(
            th.distributions.Independent(th.distributions.Beta(a, b), 1),
            [th.distributions.AffineTransform(0, 10)]
        )
        return th.distributions.MixtureSameFamily(
            th.distributions.Categorical(logits=logits),
            component_distribution,
        )
