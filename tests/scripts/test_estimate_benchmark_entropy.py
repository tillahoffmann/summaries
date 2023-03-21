from summaries.scripts import estimate_benchmark_entropy
from unittest import mock


def test_estimate_benchmark_entropy():
    with mock.patch('builtins.open') as open_, mock.patch('pickle.dump') as dump_:
        estimate_benchmark_entropy.__main__(['--num_posterior_samples=5', '100', '7', 'output.pkl'])
    open_.assert_called_once_with('output.pkl', 'wb')
    result = dump_.call_args[0][0]
    for key in ['random', 'fearnhead']:
        assert result['entropies'][key].shape == (7, 100)
        assert result['coefficients'][key].shape == (7, 6)
