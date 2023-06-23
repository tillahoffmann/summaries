import functools as ft
import logging
import numpy as np
import pytest
from summaries.base import Container, maybe_apply_to_container, ParamDict
from summaries.compressors import Compressor, LinearPosteriorMeanCompressor, \
    MinimumConditionalEntropyCompressor, NonLinearPosteriorMeanCompressor, TorchCompressor
import torch as th
from typing import Callable


th.set_default_dtype(th.float64)


def pytest_sessionfinish():
    logging.raiseExceptions = False


@pytest.fixture
def basic_params() -> Container[ParamDict]:
    n = 100_000
    m = 100
    return Container(
        {"theta": np.random.normal(0, 1, n), "dummy": np.random.normal(0, 1, n)},
        {"theta": np.random.normal(0, 1, m), "dummy": np.random.normal(0, 1, m)},
    )


@pytest.fixture
def basic_data(basic_params: Container[ParamDict]) -> Container[np.ndarray]:
    @maybe_apply_to_container
    def _sample(params: ParamDict):
        n = params["theta"].size
        p = 3
        return params["theta"][..., None] * [0, 1, 1] + np.random.normal(0, .1, (n, p))

    return _sample(basic_params)


class Mean(th.nn.Module):
    def forward(self, x: th.Tensor) -> th.Tensor:
        x = th.as_tensor(x)
        return x.mean(axis=-1, keepdim=True)


compressor_configurations = {
    "LinearPosteriorMeanCompressor": (LinearPosteriorMeanCompressor, {}),
    "NonLinearPosteriorMeanCompressor": (
        NonLinearPosteriorMeanCompressor,
        {"hidden_layer_sizes": [9, 7], "random_state": 9, "max_iter": 3}
    ),
    "MinimumConditionalEntropyCompressor": (
        MinimumConditionalEntropyCompressor,
        {"nearest_neighbor_frac": 0.001}
    ),
    "TorchCompressor": (TorchCompressor, {"compressor": Mean()}),
}


@pytest.fixture(params=compressor_configurations.values(), ids=compressor_configurations.keys())
def basic_compressor_factory(request: pytest.FixtureRequest) -> Callable[..., Compressor]:
    compressor_cls, kwargs = request.param
    return ft.partial(compressor_cls, **kwargs)
