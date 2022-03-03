import numpy as np
from scipy import special


class Distribution:
    """
    Abstract base class for distributions.
    """
    def _normalize_shape(self, shape):
        if isinstance(shape, int):
            return shape,
        elif shape is None:
            return ()
        return tuple(shape)

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def sample(self, size: tuple = None) -> np.ndarray:
        raise NotImplementedError

    @property
    def mean(self) -> np.ndarray:
        raise NotImplementedError


class NormalDistribution(Distribution):
    """
    Normal distribution.
    """
    def __init__(self, loc, scale):
        self.loc = np.asarray(loc)
        self.scale = np.asarray(scale)

    def log_prob(self, x) -> np.ndarray:
        z = (x - self.loc) / self.scale
        return - (np.log(2 * np.pi * self.scale ** 2) + z ** 2) / 2

    def sample(self, size=None) -> np.ndarray:
        size = self._normalize_shape(size) + np.broadcast_shapes(self.loc.shape, self.scale.shape)
        return np.random.normal(self.loc, self.scale, size)

    @property
    def mean(self) -> np.ndarray:
        return self.loc


class NegativeBinomialDistribution(Distribution):
    """
    Negative binomial distribution.
    """
    def __init__(self, n, p):
        self.n = np.asarray(n)
        self.p = np.asarray(p)
        self._gammaln_n = special.gammaln(self.n)
        self._logp = np.log(self.p)
        self._log1mp = np.log1p(-self.p)

    def log_prob(self, x):
        return x * self._log1mp + self.n * self._logp + special.gammaln(x + self.n) \
            - self._gammaln_n - special.gammaln(x + 1)

    def sample(self, size=None):
        size = self._normalize_shape(size) + np.broadcast_shapes(self.n.shape, self.p.shape)
        return np.random.negative_binomial(self.n, self.p, size)

    @property
    def mean(self):
        return self.n * (1 - self.p) / self.p


class UniformDistribution(Distribution):
    """
    Uniform distribution.
    """
    def __init__(self, lower, upper):
        self.lower = np.asarray(lower)
        self.upper = np.asarray(upper)

    def log_prob(self, x):
        return -np.log(self.upper - self.lower) * np.ones_like(x)

    def sample(self, size=None):
        size = self._normalize_shape(size) + np.broadcast_shapes(self.lower.shape, self.upper.shape)
        return np.random.uniform(self.lower, self.upper, size)

    @property
    def mean(self):
        return (self.upper + self.lower) / 2
