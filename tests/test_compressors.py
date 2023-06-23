import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from summaries.base import Container, ParamDict
from summaries.compressors import _PredictorCompressor, Compressor, \
    _ExhaustiveSubsetSelectionCompressor, MinimumConditionalEntropyCompressor
from typing import Callable


def test_compressors(basic_compressor_factory: Callable[..., Compressor],
                     basic_data: Container[np.ndarray], basic_params: Container[ParamDict]) -> None:
    compressor = basic_compressor_factory()
    # Use a single observed dataset for data-dependent compressors.
    if compressor.DATA_DEPENDENT:
        basic_data.observed = basic_data.observed[:1]
    compressor.fit(basic_data, basic_params)

    # Apply it to the container.
    transformed: Container[np.ndarray] = compressor.transform(basic_data)
    assert isinstance(transformed, Container)
    assert transformed.simulated.shape[0] == basic_data.simulated.shape[0]
    assert transformed.observed.shape[0] == basic_data.observed.shape[0]
    assert transformed.simulated.ndim == 2
    assert transformed.observed.ndim == 2

    # Apply individually.
    np.testing.assert_allclose(compressor.transform(basic_data.simulated), transformed.simulated)

    # Construct another compressor and fit it only to the simulated data. Then verify the
    # predictions are the same. This only works for compressors that are data-independent.
    if not compressor.DATA_DEPENDENT:
        other_compressor = basic_compressor_factory()
        other_compressor.fit(basic_data.simulated, basic_params.simulated)
        other_transformed_observed = other_compressor.transform(basic_data.observed)
        np.testing.assert_allclose(other_transformed_observed, transformed.observed)


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


def test_subset_selection_compressor() -> None:
    compressor = _ExhaustiveSubsetSelectionCompressor(0.1)
    with pytest.raises(ValueError, match="single realization"):
        compressor.fit(Container(None, np.zeros(2)), None)
