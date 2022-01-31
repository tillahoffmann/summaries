import numpy as np
from scipy import spatial
import typing


def sample_posterior(data: np.ndarray, reference: typing.Union[np.ndarray, spatial.KDTree],
                     num_samples: int = 10000) -> np.ndarray:
    """
    Draw samples from the reference table that minimise the distance to the data.

    Args:
        data: Data vector with `p` features.
        reference: Reference table with shape `(n, p)` (where `n` is the number of entries) or a
            K-d tree for efficient neighbor lookup.
        num_samples: Number of posterior samples to draw.

    Returns:
        d: Euclidean distance of each sample from the data.
        i: Indices of the samples in `reference`.
    """
    # Construct the KDtree if necessary.
    if isinstance(reference, np.ndarray):
        if reference.ndim != 2:
            reference = reference[:, None]
        reference = spatial.KDTree(reference)

    # Get the samples.
    return reference.query(data, k=num_samples, workers=4)
