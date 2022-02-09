import pytest
from summaries.scripts import generate_benchmark_data, run_inference
from unittest import mock


@pytest.mark.parametrize('algorithm', run_inference.ALGORITHMS)
def test_run_inference(algorithm: str):
    with mock.patch('builtins.open') as open_, mock.patch('pickle.dump') as dump_, \
            mock.patch('pickle.load') as load_:
        # Generate data.
        generate_benchmark_data.__main__(['200', 'some_path.pkl'])
        dump_.assert_called_once()
        (result, _), _ = dump_.call_args

        # Run inference.
        load_.return_value = result
        run_inference.__main__([
            '--seed=0',
            algorithm,
            'train.pkl',
            'test.pkl',
            '117',
            'output.pkl',
        ])
        open_.assert_called_with('output.pkl', 'wb')
        assert open_.call_count == 4
        assert dump_.call_count == 2
