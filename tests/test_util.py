import matplotlib.colors
from matplotlib import pyplot as plt
import numpy as np
import pytest
from scipy import stats
import summaries


NUM_SAMPLES = 10000


@pytest.mark.parametrize('size', [(NUM_SAMPLES, 1), NUM_SAMPLES])
def test_estimate_entropy(size):
    scale = 7
    x = np.random.normal(0, scale, size)
    actual = summaries.estimate_entropy(x)
    expected = (np.log(2 * np.pi * scale ** 2) + 1) / 2
    assert abs(actual - expected) < .1


@pytest.mark.parametrize('method', ['singh', 'kl'])
@pytest.mark.parametrize('normalize', ['x', 'y', 'xy', False])
def test_estimate_mutual_information(method, normalize):
    cov = np.asarray([[1, .7], [.7, 2]])
    x = np.random.multivariate_normal(np.zeros(2), cov, size=NUM_SAMPLES)
    actual = summaries.estimate_mutual_information(*x.T, method=method, normalize=normalize)
    expected = (np.log(np.diag(cov)).sum() - np.linalg.slogdet(cov)[1]) / 2
    if normalize == 'x':
        norm = stats.norm(0, cov[0, 0]).entropy()
    elif normalize == 'y':
        norm = stats.norm(0, cov[1, 1]).entropy()
    elif normalize == 'xy':
        norm = stats.norm(0, np.diag(cov)).entropy().mean()
    elif not normalize:
        norm = 1
    else:
        raise NotImplementedError(normalize)
    expected /= norm
    assert abs(actual - expected) < .1


def test_rmse_and_rmse_uniform():
    x, y = np.random.uniform(0, 2, (2, NUM_SAMPLES))
    actual = summaries.evaluate_rmse(x, y)
    expected = summaries.evaluate_rmse_uniform(2)
    assert abs(actual - expected) < .1


def test_mae_and_mae_uniform():
    x, y = np.random.uniform(0, 2, (2, NUM_SAMPLES))
    actual = summaries.evaluate_mae(x, y)
    expected = summaries.evaluate_mae_uniform(2)
    assert abs(actual - expected) < .1


def test_evaluate_level():
    dist = stats.norm(0, 1)
    lin = np.linspace(-3, 3, 1000)
    density = dist.pdf(lin)
    level = summaries.evaluate_credible_level(density, 0.0455)
    assert abs(level - dist.pdf(2)) < 1e-2


@pytest.mark.parametrize('label_offset', [None, 2])
@pytest.mark.parametrize('offset', [0.05, (0.02, 0.03)])
def test_label_axes(label_offset, offset):
    # Just make sure we can run the code but minimal verification.
    _, axes = plt.subplots(2, 3)
    elements = summaries.label_axes(axes.ravel(), label_offset=label_offset, offset=offset)
    assert len(elements) == 6


def test_trapznd():
    lins = [np.linspace(0, 1, n) for n in [100, 101]]
    xx = np.asarray(np.meshgrid(*lins))
    dist = stats.beta([3, 2], [4, 5])
    pdf = np.prod(dist.pdf(xx.T).T, axis=0)
    assert pdf.shape == (101, 100)
    norm = summaries.trapznd(pdf, *lins)
    assert abs(norm - 1) < 1e-3


@pytest.mark.parametrize('p', [1, 3])
def test_whiten_features(p):
    cov = np.atleast_2d(stats.wishart(10, np.eye(p)).rvs())
    x = np.random.multivariate_normal(np.zeros(p), cov, 1000)
    y = summaries.whiten_features(x)
    np.testing.assert_allclose(np.cov(y, rowvar=False), np.eye(p), atol=1e-9)


def test_alpha_cmap():
    cmap = summaries.util.alpha_cmap(0)
    levels = [0., .3, .7, 1.]
    for level in levels:
        color = cmap(level)
        assert color[:-1] == matplotlib.colors.to_rgb('C0')
        assert abs(color[-1] - level) < 0.01
