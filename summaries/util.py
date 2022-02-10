import matplotlib.axes
import matplotlib.colors
import numpy as np
from scipy import spatial, special
import string
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


def label_axes(axes: list[matplotlib.axes.Axes], labels: list[str] = None, loc: str = 'top left',
               offset: float = 0.05, label_offset: int = None, **kwargs):
    """
    Add labels to axes.

    Args:
        axes: Iterable of matplotlib axes.
        labels: Iterable of labels (defaults to lowercase letters in parentheses).
        loc: Location of the label as a string (defaults to top left).
        offset: Offset for positioning labels in axes coordinates.
        label_offset: Index by which to offset labels.
    """
    if labels is None:
        labels = [f'({x})' for x in string.ascii_lowercase]
    if label_offset is not None:
        labels = labels[label_offset:]
    if isinstance(offset, float):
        xfactor = yfactor = offset
    else:
        xfactor, yfactor = offset
    y, x = loc.split()
    kwargs = {'ha': x, 'va': y} | kwargs
    xloc = xfactor if x == 'left' else (1 - xfactor)
    yloc = yfactor if y == 'bottom' else (1 - yfactor)
    elements = []
    for ax, label in zip(axes, labels):
        elements.append(ax.text(xloc, yloc, label, transform=ax.transAxes, **kwargs))
    return elements


def trapznd(y, *xs, axis=-1):
    """
    Evaluate a multivariate integral using the trapezoidal rule.

    Args:
        y: Integrand.
        xs: Sample points corresponding to the integrand.
    """
    for x in xs:
        y = np.trapz(y, x, axis=axis)
    return y


def evaluate_credible_level(density: np.ndarray, alpha: float) -> float:
    r"""
    Evaluate the level :math:`t` such that the density exceeding :math:`t` integrates to
    :math:`1 - \alpha`, i.e.

    .. math::

        1 - \alpha = \int dx p(x) \left[p(x) > t\right]

    Args:
        density: Density of a distribution.
        alpha: Significance level.
    """
    density = density.ravel()
    density = density[np.argsort(-density)]
    cum = np.cumsum(density)
    i = np.argmax(cum > (1 - alpha) * cum[-1])
    return density[i]


def whiten_features(features: np.ndarray) -> np.ndarray:
    """
    Whiten the features such that they have zero mean and identity variance.
    """
    features = features - features.mean(axis=0, keepdims=True)
    cov = np.atleast_2d(np.cov(features, rowvar=False))
    cholesky = np.linalg.cholesky(cov)
    inverse_cholesky = np.linalg.inv(cholesky)
    return features @ inverse_cholesky.T


def alpha_cmap(color, name: str = None, **kwargs) -> matplotlib.colors.Colormap:
    """
    Create a monochrome colormap that maps scalars to varying transparencies.

    Args:
        color : Base color to use for the colormap.
        name : Name of the colormap.
        **kwargs : dict
        Keyword arguments passed to :meth:`mpl.colors.LinearSegmentedColormap.from_list`.
    Returns
    -------
    cmap : mpl.colors.Colormap
        Colormap encoding scalars as transparencies.
    """
    if isinstance(color, int):
        color = f'C{color}'
    name = name or f'alpha_cmap_{color}'
    return matplotlib.colors.LinearSegmentedColormap.from_list(name, [
        matplotlib.colors.to_rgba(color, alpha=0.0),
        matplotlib.colors.to_rgba(color, alpha=1.0),
    ])
