import numpy as np
from scipy import spatial
import typing


class Algorithm:
    def sample_posterior(self, data: np.ndarray, num_samples: int, **kwargs) -> np.ndarray:
        raise NotImplementedError

    @property
    def num_params(self):
        raise NotImplementedError


class RejectionAlgorithm(Algorithm):
    """
    Simple approximate Bayesian computation rejection sampler with optional feature whitening.

    Args:
        train_features: Reference table of simulated data with shape `(n, p)`, where `n` is the
            number of simulations and `p` is the number of features.
        train_params: Reference table of simulated parameter values with shape `(n, q)`, where `n`
            is the number of simulatiosn and `q` is the number of parameters.
        whiten_features: Whether to whiten the features by multiplying them by the inverse Cholesky
            decomposition of the covariance matrix. This is equivalent to using the Mahalanobis
            distance.
        transform: Optional transform to apply to the features.
    """
    def __init__(self, train_features: np.ndarray, train_params: np.ndarray,
                 whiten_features: bool = False, transform: typing.Callable = None):
        # Validate inputs.
        self.transform = transform
        if self.transform:
            train_features = self.transform(train_features)
        self.train_features = np.asarray(train_features)
        assert self.train_features.ndim == 2, 'expected features to have two dimensions but got ' \
            f'shape {self.train_features.shape}'

        self.train_params = np.asarray(train_params)
        assert self.train_params.ndim == 2, 'expected parameters to have two dimensions but got ' \
            f'shape {self.train_params.shape}'
        assert self.train_features.shape[0] == self.train_params.shape[0]

        if whiten_features:
            cov = np.atleast_2d(np.cov(self.train_features, rowvar=False))
            chol = np.linalg.cholesky(cov)
            self.inverse_cholesky = np.linalg.inv(chol)
            features = self.train_features @ self.inverse_cholesky
            # Check that the transformation whitened the features.
            np.testing.assert_allclose(np.cov(features, rowvar=False), np.eye(cov.shape[0]))
        else:
            self.inverse_cholesky = None
            features = self.train_features

        self.reference = spatial.KDTree(features)

    def sample_posterior(self, features: np.ndarray, num_samples: int, return_indices: bool = False,
                         return_distances: bool = False, **kwargs) -> np.ndarray:
        """
        Draw samples from the reference table that minimise the distance to the data.

        Args:
            features: Data vector with `p` features.
            num_samples: Number of posterior samples to draw.
            return_indices: Whether to return indices in the reference table.
            return_distances: Whether to return distances between the data and elements of the
                reference table.

        Returns:
            y: Posterior samples.
            i: Indices of the samples in `reference`.
            d: Distance of each sample from the data (if `return_distances` is `True`).
        """
        if self.transform:
            features = self.transform(features)
        features = np.asarray(features)
        if self.inverse_cholesky is not None:
            features = features @ self.inverse_cholesky

        distances, indices = self.reference.query(features, k=num_samples, **kwargs)
        y = self.train_params[indices]
        if not (return_indices or return_distances):
            return y
        y = [y]
        if return_indices:
            y.append(indices)
        if return_distances:
            y.append(distances)
        return tuple(y)

    @property
    def num_params(self):
        return self.train_params.shape[1]
