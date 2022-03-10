import matplotlib.figure
from summaries import benchmark
from summaries.examples import bimodal, broad_posterior, piecewise_likelihood
from summaries.scripts.plot import __main__
import pytest
from unittest import mock


def test_plot_script():
    with mock.patch('matplotlib.figure.Figure.savefig') as savefig_:
        __main__(['summaries.examples.bimodal:_plot_example', '--style=scrartcl.mplstyle',
                  'output.pdf'])
    savefig_.assert_called_once_with('output.pdf')


@pytest.mark.parametrize('func', [bimodal._plot_example, broad_posterior._plot_example,
                                  piecewise_likelihood._plot_example, benchmark._plot_example])
def test_plots(func):
    assert isinstance(func(), matplotlib.figure.Figure)
