import numpy as np
from scipy import spatial, special
import typing


def estimate_entropy(x: np.ndarray, k: int = 4, method: str = 'singh') -> float:
    """
    Estimate the entropy of a point cloud.

    Args:
        x: Coordinate of points.
        k: Nearest neighbor to use for entropy estimation.
        method: Method used for entropy estimation. See 10.1080/01966324.2003.10737616 for
            :code:`singh` and ??? for :code:`kl`.

    Returns:
        entropy: Estimated entropy of the point cloud.
    """
    # Ensure the point cloud has the right shape.
    x = maybe_add_batch_dim(x)
    n, p = x.shape

    # Use a KD tree to look up the k^th nearest neighbour.
    tree = spatial.KDTree(x)
    distance, _ = tree.query(x, k=k + 1)
    distance = distance[:, -1]

    # Estimate the entropy.
    if method == 'singh':
        return p * np.log(np.pi) / 2 - special.gammaln(p / 2 + 1) - special.digamma(k) + np.log(n) \
            + p * np.log(distance).mean()
    elif method == 'kl':
        return special.digamma(n) - special.digamma(k) + p * np.log(np.pi) / 2 \
            - special.gammaln(p / 2 + 1) + p * np.log(distance).mean()
    else:
        raise NotImplementedError(method)


def estimate_mutual_information(
        x: np.ndarray, y: np.ndarray, normalize: typing.Union[bool, str] = False,
        method: str = 'singh') -> float:
    """
    Estimate the mutual information between two variables.

    Args:
        x: First variable.
        y: Second variable.
        normalize: Whether to normalize the mutual information. Use `x` to divide by entropy of
            first variable, `y` to divide by entropy of second variable, or `xy` to divide by
            mean entropy of `x` and `y`.
        method: Nearest neighbor method to use for entropy estimation.

    Returns:
        mutual_information: Mutual information estimate (possibly normalised).
    """
    x = maybe_add_batch_dim(x)
    y = maybe_add_batch_dim(y)
    entropy_x = estimate_entropy(x, method=method)
    entropy_y = estimate_entropy(y, method=method)
    entropy_xy = estimate_entropy(np.hstack([x, y]), method=method)
    mi = entropy_x + entropy_y - entropy_xy
    if normalize == 'x':
        mi /= entropy_x
    elif normalize == 'y':
        mi /= entropy_y
    elif normalize == 'xy':
        mi /= (entropy_x + entropy_y) / 2
    elif normalize:
        raise NotImplementedError(normalize)
    return mi


def evaluate_rmse(x: np.ndarray, y: np.ndarray = None, axis=None) -> np.ndarray:
    """
    Evaluate the root mean squared error.
    """
    if y is not None:
        x = x - y
    return np.sqrt(np.square(x).mean(axis=axis))


def evaluate_mae(x: np.ndarray, y: np.ndarray = None, axis=None) -> np.ndarray:
    """
    Evaluate the root mean squared error.
    """
    if y is not None:
        x = x - y
    return np.abs(x).mean(axis=axis)


def evaluate_rmse_uniform(interval):
    """
    Evaluate the expected root mean squared error when both the true value and the estimate are
    drawn uniformly from a given interval.
    """
    return interval / np.sqrt(6)


def evaluate_mae_uniform(interval):
    """
    Evaluate the expected mean absolute error when both the true value and the estimate are drawn
    uniformly from a given interval.
    """
    return interval / 3


def maybe_add_batch_dim(x: np.ndarray) -> np.ndarray:
    """
    Add a batch dimension to the array if required. If :attr:`x` has shape :code:`(n,)` the returned
    value will have shape :code:`(n, 1)`. If :attr`x` has more than one dimension, it will be
    returned unchanged.

    Args:
        x: Array to add a batch dimension to if required.

    Returns:
        x: Array with batch dimension added if required.
    """
    x = np.asarray(x)
    if x.ndim > 1:
        return x

    return x[:, None]
