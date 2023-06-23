import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from summaries.base import Container
from summaries.compressors import _PredictorCompressor, Compressor, LinearPosteriorMeanCompressor, \
    MinimumConditionalEntropyCompressor, NonLinearPosteriorMeanCompressor, TorchCompressor
import torch as th
from typing import Dict, Type


class Mean(th.nn.Module):
    def forward(self, x: th.Tensor) -> th.Tensor:
        return x.mean(axis=-1)


@pytest.mark.parametrize("compressor_cls, kwargs", [
    (LinearPosteriorMeanCompressor, {}),
    (NonLinearPosteriorMeanCompressor,
     {"hidden_layer_sizes": [9, 7], "random_state": 9, "max_iter": 3}),
    (MinimumConditionalEntropyCompressor, {"nearest_neighbor_frac": 0.001}),
    (TorchCompressor, {"module": Mean()}),
])
def test_compressors(compressor_cls: Type[Compressor], kwargs: Dict) -> None:
    # Simulate some data.
    n = 100_000
    m = 10
    p = 3

    theta = np.random.normal(0, 1, n + m)
    x = theta[:, None] + np.random.normal(0, 1, (theta.size, p))
    data = Container(x[m:], x[:m])
    params = Container(
        {"theta": theta[m:], "_": np.random.normal(0, 1, n)},
        {"theta": theta[:m], "_": np.random.normal(0, 1, m)},
    )

    # Construct a compressor and fit it to the full data.
    compressor = compressor_cls(**kwargs)
    compressor.fit(data, params)

    # Apply it to the container.
    transformed = compressor.transform(data)
    assert isinstance(transformed, Container)
    assert transformed.simulated.shape[0] == n
    assert transformed.observed.shape[0] == m

    # Apply individually.
    np.testing.assert_allclose(compressor.transform(data.simulated), transformed.simulated)

    # Construct another compressor and fit it only to the simulated data. Then verify the
    # predictions are the same. This only works for compressors that are data-independent.
    if not isinstance(compressor, MinimumConditionalEntropyCompressor):
        other_compressor = compressor_cls(**kwargs)
        other_compressor.fit(data.simulated, params.simulated)
        np.testing.assert_allclose(other_compressor.transform(data.observed), transformed.observed)


def test_data_dependent_compressor_invalid() -> None:
    compressor = MinimumConditionalEntropyCompressor(0.1)
    with pytest.raises(ValueError, match="must be a `Container`"):
        compressor.fit(None, None)

    with pytest.raises(NotFittedError):
        compressor.transform(None)


def test_predictor_compressor_invalid() -> None:
    compressor = _PredictorCompressor(LinearRegression(), "foo")
    with pytest.raises(ValueError, match="does not support"):
        compressor.transform(None)
