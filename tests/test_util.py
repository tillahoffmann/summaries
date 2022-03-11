import matplotlib.colors
from matplotlib import pyplot as plt
from unittest import mock
import numpy as np
import pytest
from scipy import stats
import summaries
import torch as th


NUM_SAMPLES = 10000


@pytest.mark.parametrize('size', [(NUM_SAMPLES, 1), NUM_SAMPLES])
def test_estimate_entropy(size):
    scale = 7
    dist = th.distributions.Normal(0, scale)
    x = dist.sample(size if isinstance(size, tuple) else (size,))
    actual = summaries.estimate_entropy(x)
    expected = dist.entropy()
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
    x, y = th.distributions.Uniform(0, 2).sample((2, NUM_SAMPLES))
    actual = summaries.evaluate_rmse(x, y)
    expected = summaries.evaluate_rmse_uniform(2)
    assert abs(actual - expected) < .1


def test_mae_and_mae_uniform():
    x, y = th.distributions.Uniform(0, 2).sample((2, NUM_SAMPLES))
    actual = summaries.evaluate_mae(x, y)
    expected = summaries.evaluate_mae_uniform(2)
    assert abs(actual - expected) < .1


def test_evaluate_level():
    dist = th.distributions.Normal(0, 1)
    lin = th.linspace(-3, 3, 1000)
    density = dist.log_prob(lin).exp()
    level = summaries.evaluate_credible_level(density, 0.0455)
    assert abs(level - dist.log_prob(th.scalar_tensor(2.0)).exp()) < 1e-2


@pytest.mark.parametrize('config', [
    {'shape': (), 'labels': None, 'label_offset': None, 'offset': 0.05},
    {'shape': (), 'labels': 'single', 'label_offset': None, 'offset': (0.02, 0.03)},
    {'shape': (1, 2), 'labels': ['list1', 'list2'], 'label_offset': None, 'offset': (0.02, 0.03)},
    {'shape': (2, 3), 'labels': None, 'label_offset': 3, 'offset': 0.05}
])
def test_label_axes(config: dict):
    num_elements = np.prod(config['shape'])
    _, axes = plt.subplots(*config['shape'])
    if num_elements > 1:
        axes = axes.ravel()
    elements = summaries.label_axes(axes, labels=config['labels'], offset=config['offset'],
                                    label_offset=config['label_offset'])
    assert len(elements) == num_elements


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


@pytest.mark.parametrize('p', [1, 2])
def test_estimate_divergence(p):
    n = 100000
    m = 100000
    loc1 = 0
    loc2 = 1.4
    scale1 = 0.7
    scale2 = 1.3
    var1 = scale1 ** 2
    var2 = scale2 ** 2
    expected = p * ((loc1 - loc2) ** 2 / var2 + var1 / var2 - 1 - np.log(var1 / var2)) / 2
    x = th.distributions.Normal(loc1, scale1).sample((n, p))
    y = th.distributions.Normal(loc2, scale2).sample((m, p))
    actual = summaries.estimate_divergence(x, y, k=7)
    assert abs(expected - actual) < 0.05


@pytest.mark.parametrize('func', [None, th.as_tensor])
def test_transpose_samples(func):
    samples = [{'x': x} for x in th.arange(10)]
    samples = summaries.transpose_samples(samples, func=func)
    assert 'x' in samples
    if func:
        assert isinstance(samples['x'], th.Tensor)
    else:
        assert isinstance(samples['x'], list)


def test_normalize_shape():
    assert summaries.normalize_shape(None) == ()
    assert summaries.normalize_shape(5) == (5,)
    assert summaries.normalize_shape((4, 5)) == (4, 5)


def test_setup_with_seed():
    with mock.patch('os.environ.get') as env_get, mock.patch('numpy.random.seed') as np_seed, \
            mock.patch('torch.manual_seed') as th_seed, mock.patch('logging.basicConfig') as config:
        env_get.side_effect = ('debug', 23)
        summaries.setup_script()
        config.assert_called_once_with(level='DEBUG')
        np_seed.assert_called_once_with(23)
        th_seed.assert_called_once_with(23)
