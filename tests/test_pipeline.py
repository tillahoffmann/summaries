import numpy as np
from sklearn.pipeline import Pipeline
from summaries.algorithms import NearestNeighborAlgorithm
from summaries.base import Container, ParamDict
from summaries.compressors import Compressor, MinimumConditionalEntropyCompressor
from typing import Callable


def test_pipeline(basic_compressor_factory: Callable[..., Compressor],
                  basic_data: Container[np.ndarray], basic_params: Container[ParamDict]) -> None:
    compressor = basic_compressor_factory()
    # Use a single observed dataset for data-dependent compressors.
    if isinstance(compressor, MinimumConditionalEntropyCompressor):
        basic_data.observed = basic_data.observed[:1]

    m = basic_data.observed.shape[0]

    nearest_neighbor = NearestNeighborAlgorithm(0.01)
    pipeline = Pipeline([
        ("compress", compressor),
        ("sample", nearest_neighbor)
    ])

    pipeline.fit(basic_data, basic_params)
    samples: ParamDict = pipeline.predict(basic_data.observed)
    assert {key: value.shape for key, value in samples.items()} \
        == {"dummy": (m, 1000), "theta": (m, 1000)}
