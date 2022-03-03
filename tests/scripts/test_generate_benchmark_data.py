from summaries.scripts import generate_benchmark_data
from unittest import mock


def test_generate_benchmark_data():
    with mock.patch('builtins.open') as open_, mock.patch('pickle.dump') as dump_:
        generate_benchmark_data.__main__(['--seed=0', '23', 'some_file.pkl'])

    open_.assert_called_once_with('some_file.pkl', 'wb')
    dump_.assert_called_once()
    (result, _), _ = dump_.call_args
    assert 'args' in result
    assert len(result['samples']) == 23
