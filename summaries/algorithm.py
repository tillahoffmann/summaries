import numpy as np
from scipy import spatial


class Algorithm:
    def sample_posterior(self, data: np.ndarray, num_samples: int, **kwargs) -> np.ndarray:
        raise NotImplementedError


class RejectionAlgorithm(Algorithm):
    """
    Simple approximate Bayesian computation rejection sampler with optional feature whitening.

    Args:
        train: Reference table of simulations with shape `(n, p)`, where `n` is the number of
            simulations and `p` is the number of features.
        whiten_features: Whether to whiten the features by multiplying them by the inverse Cholesky
            decomposition of the covariance matrix. This is equivalent to using the Mahalanobis
            distance.
    """
    def __init__(self, train: np.ndarray, whiten_features: bool = False):
        # Transform to the right shape.
        self.train = np.asarray(train)
        if self.train.ndim == 1:
            self.train = self.train[:, None]

        if whiten_features:
            cov = np.atleast_2d(np.cov(self.train, rowvar=False))
            chol = np.linalg.cholesky(cov)
            self.inverse_cholesky = np.linalg.inv(chol)
            features = self.train @ self.inverse_cholesky
            # Check that the transformation whitened the features.
            np.testing.assert_allclose(np.cov(features, rowvar=False), np.eye(cov.shape[0]))
        else:
            self.inverse_cholesky = None
            features = self.train

        self.reference = spatial.KDTree(features)

    def sample_posterior(self, data: np.ndarray, num_samples: int, return_distances: bool = False,
                         **kwargs) -> np.ndarray:
        """
        Draw samples from the reference table that minimise the distance to the data.

        Args:
            data: Data vector with `p` features.
            num_samples: Number of posterior samples to draw.
            return_indices: Whether to return indices in the reference table.
            return_distances: Whether to return distances between the data and elements of the
                reference table.

        Returns:
            i: Indices of the samples in `reference`.
            d: Distance of each sample from the data (if `return_distances` is `True`).
        """
        if self.inverse_cholesky is not None:
            data = data @ self.inverse_cholesky
        distance, i = self.reference.query(data, k=num_samples, **kwargs)
        return (i, distance) if return_distances else i
