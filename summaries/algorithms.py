from __future__ import annotations
import numpy as np
from scipy.spatial import KDTree
from sklearn.exceptions import NotFittedError
from typing import Optional

from .base import maybe_apply_to_container, maybe_apply_to_simulated, ParamDict


class NearestNeighborAlgorithm:
    """
    Draw approximate posterior samples using a nearest neighbor algorithm.

    Args:
        frac: Fraction of samples to return as approximate posterior samples.
        minkowski_norm: Minkowski p-norm to use for queries (defaults to Euclidean distances).
        **kdtree_kwargs: Keyword arguments passed to the KDTree constructor.
    """
    def __init__(self, frac: float, minkowski_norm: float = 2, **kdtree_kwargs) -> None:
        self.frac = frac
        self.minkowski_norm = minkowski_norm
        self.kdtree_kwargs = kdtree_kwargs

        self.tree_: Optional[KDTree] = None
        self.params_: Optional[ParamDict] = None

    @maybe_apply_to_simulated
    def fit(self, data: np.ndarray, params: ParamDict) -> NearestNeighborAlgorithm:
        """
        Construct a :class:`.KDTree` for fast nearest neighbor search for sampling parameters.

        Args:
            data: Simulated data or summary statistics used to build the tree.
            params: Dictionary of parameters used to generate the corresponding `data` realization.
        """
        self.tree_ = KDTree(data, **self.kdtree_kwargs)
        self.params_ = params
        return self

    @maybe_apply_to_container
    def predict(self, data: np.ndarray) -> ParamDict:
        """
        Draw approximate posterior samples.

        Args:
            data: Data to condition on with shape `(batch_size, num_features)`.

        Returns:
            Dictionary of posterior samples. Each value has shape
            `(batch_size, num_samples, *event_shape)`, where `event_shape` is the basic shape of the
            corresponding parameter.
        """
        # Validate the state and input arguments.
        if self.tree_ is None:
            raise NotFittedError

        data = np.asarray(data)
        if data.ndim != 2:
            raise ValueError(f"data must be a matrix; got shape {data.shape}")
        if data.shape[1] != self.tree_.m:
            raise ValueError(f"data must have {self.tree_.m} features; got {data.shape[1]}")

        num_samples = int(self.frac * self.tree_.n)
        _, idx = self.tree_.query(data, k=num_samples, p=self.minkowski_norm)

        return {key: value[idx] for key, value in self.params_.items()}
