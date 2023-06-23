from __future__ import annotations
import itertools as it
import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import torch as th
from tqdm import tqdm
from typing import Any, Optional

from .algorithms import NearestNeighborAlgorithm
from .base import Container, ContainerOrValue, maybe_apply_to_container, maybe_apply_to_simulated, \
    ParamDict, ravel_param_dict
from .util import estimate_entropy


class Compressor:
    """
    Compress data to summary statistics.

    Data-dependent compressors, such as the conditional posterior entropy minimization of Nunes and
    Balding (2010), have to be fit for each observed datasets. Data-independent compressors only use
    simulations to learn informative summary statistics.

    The implementation is
    `duck-typed <https://scikit-learn.org/stable/glossary.html#term-duck-typing>`__ to match a
    `scikit-learn transformer <https://scikit-learn.org/stable/glossary.html#term-transformer`__.
    """
    def transform(self, data: ContainerOrValue) -> ContainerOrValue:
        """
        Transform (simulated) data to summary statistics.

        Args:
            data: Data to transform.

        Returns:
            Summary statistics.
        """
        raise NotImplementedError

    def fit(self, data: ContainerOrValue, params: ContainerOrValue[ParamDict]) -> Compressor:
        """
        Fit the compressor to data and parameters.

        Args:
            simulated: Data to fit.
            observed: Parameters to fit.

        Returns:
            The compressor.
        """
        raise NotImplementedError


class _ExhaustiveSubsetSelectionCompressor(Compressor):
    """
    Select a subset of summary statistics based on a loss function.

    Args:
        **algorithm_kwargs: Keyword arguments passed to the sampling algorithm.
    """
    def __init__(self, nearest_neighbor_frac: float, show_progress: bool = True,
                 **nearest_neighbor_kwargs) -> None:
        self.nearest_neighbor_frac = nearest_neighbor_frac
        self.show_progress = show_progress
        self.nearest_neighbor_kwargs = nearest_neighbor_kwargs
        self.masks_: Optional[np.ndarray] = None
        self.losses_: Optional[np.ndarray] = None

    def fit(self, data: Container[np.ndarray], params: Container[np.ndarray]) -> Compressor:
        if not isinstance(data, Container):
            raise ValueError("data must be a `Container` for data-dependent subset selection "
                             f"algorithms; got {data}")
        if data.observed.ndim != 2 or data.observed.shape[0] != 1:
            raise ValueError("observed data must be a single realization for data-dependent subset "
                             f"selection algorithms; got shape {data.observed.shape}")
        _, num_features = data.simulated.shape
        self.masks_ = np.asarray([mask for mask in it.product(*[(False, True)] * num_features)
                                  if any(mask)])
        self.losses_ = np.asarray([
            self._evaluate_mask(data, params, mask) for mask in
            (tqdm(self.masks_) if self.show_progress else self.masks_)
        ])
        return self

    def _evaluate_mask(self, data: Container[np.ndarray], params: Container[np.ndarray],
                       mask: np.ndarray) -> float:
        """
        Evaluate a candidate mask.

        Args:
            simulated: Reference table comprising parameters and simulated data.
            observed: Observed data.
            mask: Mask to evaluate.
        """
        algorithm = NearestNeighborAlgorithm(self.nearest_neighbor_frac,
                                             **self.nearest_neighbor_kwargs)
        algorithm.fit(data.simulated[:, mask], params.simulated)
        samples = algorithm.predict(data.observed[:, mask])
        return self._evaluate_loss(samples)

    def _evaluate_loss(self, samples: ParamDict) -> np.ndarray:
        """
        Evaluate a loss for candidate posterior samples.

        Args:
            samples: Posterior samples to evaluate.

        Returns:
            Vector of loss values.
        """
        raise NotImplementedError

    @property
    def best_mask_(self) -> np.ndarray:
        """
        Best mask with the smallest loss.
        """
        if self.losses_ is None:
            raise NotFittedError
        return self.masks_[np.argmin(self.losses_)]

    @maybe_apply_to_container
    def transform(self, data: np.ndarray) -> np.ndarray:
        return data[:, self.best_mask_]


class MinimumConditionalEntropyCompressor(_ExhaustiveSubsetSelectionCompressor):
    """
    Select summary statistics to minimize the conditional entropy of posterior samples as proposed
    by Nunes and Balding (2010).
    """
    def _evaluate_loss(self, samples: ParamDict) -> np.ndarray:
        # Ravel the parameters for which we want to infer entropy. We maintain two batch dimensions:
        # one of size one for the single sample and one for the number of samples drawn.
        raveled = ravel_param_dict(samples, 2)
        assert raveled.shape[0] == 1
        return estimate_entropy(raveled[0])


class SklearnEstimator:
    """
    Type stub for scikit-learn estimators that can be fit to data.

    There are no type stubs for scikit-learn (cf
    https://github.com/scikit-learn/scikit-learn/issues/16705).
    """
    def fit(self, X: Any, y: Any) -> SklearnEstimator:
        ...  # pragma: no cover


class _PredictorCompressor:
    """
    Compress with a scikit-learn predictor for data-independent compression.

    The predictor (such as LinearRegression and MLPRegression for continuous parameters or
    LogisticRegression for binary parameters) is fit using simulated data as features and the
    parameters used to simulate the data as targets. The parameters are "raveled" to before calling
    `fit` (see :func:`.base.ravel_param_dict` for details).

    Args:
        predictor: Prediction algorithm for parameters.
        method: Prediction method, such as `predict`, `predict_proba`, or `predict_log_proba`.
    """
    def __init__(self, predictor: SklearnEstimator, method: str) -> None:
        self.predictor = predictor
        self.method = method

    @maybe_apply_to_simulated
    def fit(self, data: np.ndarray, params: ParamDict) -> _PredictorCompressor:
        self.predictor.fit(data, ravel_param_dict(params, 1))
        return self

    @maybe_apply_to_container
    def transform(self, data: np.ndarray) -> np.ndarray:
        try:
            method = getattr(self.predictor, self.method)
        except AttributeError as ex:
            raise ValueError(f"predictor {self.predictor} does not support method {self.method}") \
                from ex
        return method(data)


class LinearPosteriorMeanCompressor(_PredictorCompressor):
    """
    Estimate the posterior mean using linear regression as proposed by Fearnhead and Prangle (2012).
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(LinearRegression(**kwargs), "predict")


class NonLinearPosteriorMeanCompressor(_PredictorCompressor):
    """
    Estimate the posterior mean using non-linear regression as proposed by Jiang et al. (2017).
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(MLPRegressor(**kwargs), "predict")


class TorchCompressor(Compressor):
    """
    Compress with a pre-trained neural network (calling `fit` is a no-op).

    Args:
        module: Neural network that compresses the data.
    """
    def __init__(self, module: th.nn.Module) -> None:
        self.module = module

    def fit(self, data: ContainerOrValue, params: ContainerOrValue[ParamDict]) -> Compressor:
        return self

    @maybe_apply_to_container
    def transform(self, data: Any) -> th.Tensor:
        return self.module(data)
