import torch as th
import typing
from . import algorithm


class DenseStack(th.nn.Module):
    """
    Apply a sequence of dense layers.

    Args:
        num_nodes: Sequence of number of hidden nodes. The first number of nodes must match the
            input.
        activation: Activation function.
    """
    def __init__(self, num_nodes: typing.Iterable[int], activation: th.nn.Module):
        super().__init__()
        layers = []
        for i, j in zip(num_nodes, num_nodes[1:]):
            layers.extend([th.nn.Linear(i, j), activation])
        self.layers = th.nn.Sequential(*layers[:-1])

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.layers(x)


class DenseCompressor(DenseStack):
    """
    Compress a batch of observations with dimensionality :math:`p` to a batch of features with
    dimensionality :math:`q` using fully-connected layers followed by mean-pooling for each element
    of the batch. I.e. we compress a tensor of observations with shape
    :code:`(*batch_shape, num_observations, p)` to a feature tensor with shape
    :code:`(*batch_shape, q)`.

    Args:
        num_nodes: Sequence of number of nodes in each hidden layer. The first number of nodes must
            match the dimensionality of the observations, and the last number of nodes is the number
            of features.
        activation: Activation function between each fully-connected layer.
    """
    def forward(self, x: th.Tensor) -> th.Tensor:
        return super().forward(x).mean(axis=-2)


class NeuralCompressorNearestNeighborAlgorithm(algorithm.StaticCompressorNearestNeighborAlgorithm):
    """
    Compress the data using a neural network.
    """
    def __init__(self, train_data: th.Tensor, train_params: th.Tensor, path: str) -> None:
        self.model_path = path
        self.model: th.nn.Module = th.load(path)
        self.model.eval()
        super().__init__(train_data, train_params, self._target)

    def _target(self, x):
        x = th.as_tensor(x)
        with th.no_grad():
            return self.model(x)


class NeuralDensityAlgorithm(algorithm.Algorithm):
    """
    Draw samples from the posterior using a distribution parameterized by a neural network.
    """
    def __init__(self, path: str) -> None:
        self.model_path = path
        self.model: th.nn.Module = th.load(path)
        self.model.eval()

    def sample(self, data: th.Tensor, num_samples: int, show_progress: bool = True, **kwargs) \
            -> typing.Tuple[th.Tensor, dict]:
        data = th.as_tensor(data)
        with th.no_grad():
            dist: th.distributions.Distribution = self.model(data)

        # Draw a sample and move the sample batch dimension to the second to last position.
        sample: th.Tensor = dist.sample([num_samples])
        sample = sample.moveaxis(0, -2)
        return sample.numpy(), None
