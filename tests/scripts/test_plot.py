from summaries.scripts.plot import __main__
from unittest import mock


def test_plot():
    with mock.patch('matplotlib.figure.Figure.savefig') as savefig_:
        __main__([
            'summaries.examples.bimodal:_plot_example',
            '--seed=0',
            '--style=scrartcl.mplstyle',
            'output.pdf',
        ])

    savefig_.assert_called_once_with('output.pdf')
