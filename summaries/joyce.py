import numpy as np


def evaluate_joyce_score(x: np.ndarray, y: np.ndarray, epsilon: float = 1e-6,
                         aggregate: bool = True) -> np.ndarray:
    """
    Evaluate the z-score proposed by Joyce and Marjoram to assess whether two samples were drawn
    from the same distribution.

    Args:
        x: Number of observations in each histogram bin for the current set of summary statistics.
        y: Number of observations in each histogram bin for the proposed set of summary statistics.
        epsilon: Small constant to stabilise the computation for empty histogram bins.
        aggregate: Whether to aggregate the score by taking the maxium over bins.
    """
    nx = x.sum(axis=-1, keepdims=True)
    ny = y.sum(axis=-1, keepdims=True)
    expected_proba = (x + epsilon) / nx

    # Evaluate the expected value, the standard deviation, and z-score for each bin.
    expected_y = ny * expected_proba
    std_y = np.sqrt(ny * expected_proba * (1 - expected_proba))
    score = (y - expected_y) / std_y
    if aggregate:
        score = np.max(score, axis=-1)
    return score


def evaluate_log_likelihood_ratio(x: np.ndarray, y: np.ndarray, epsilon: float = 1e-6) \
        -> np.ndarray:
    r"""
    Evaluate twice the log likelihood ratio for two hypotheses: that x and y come from the same
    multinomial distribution and from different multinomial distributions. The test statistic should
    follow a :math:`\chi^2` distribution with :math:`k - 1` degrees of freedom, where :math:`k` is
    the number of histogram bins.

    Args:
        x: Number of observations in each histogram bin for the current set of summary statistics.
        y: Number of observations in each histogram bin for the proposed set of summary statistics.
        epsilon: Small constant to stabilise the computation for empty histogram bins.
    """
    # Evaluate the likelihood under the assumption that x and y come from the same distribution.
    nx = x.sum(axis=-1, keepdims=True)
    ny = y.sum(axis=-1, keepdims=True)
    p0 = (x + y + epsilon) / (nx + ny)
    ll0 = np.sum((x + y) * np.log(p0), axis=-1)

    # Evaluate the likelihood under the assumption that x and y come from different distributions.
    px = (x + epsilon) / nx
    py = (y + epsilon) / ny
    ll1 = np.sum(x * np.log(px) + y * np.log(py), axis=-1)
    return 2 * (ll1 - ll0)
