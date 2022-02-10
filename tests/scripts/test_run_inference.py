import pytest
from summaries.scripts import generate_benchmark_data, run_inference
from unittest import mock


@pytest.mark.parametrize('algorithm', run_inference.ALGORITHMS)
def test_run_inference(algorithm: str):
    num_train = 20
    num_test = 7
    num_samples = 13

    assert num_train >= num_samples, 'cannot take more samples than there are training points'

    with mock.patch('builtins.open') as open_, mock.patch('pickle.dump') as dump_, \
            mock.patch('pickle.load') as load_:
        # Generate data and mock loading (https://stackoverflow.com/a/23207767/1150961).
        generate_benchmark_data.__main__([str(num_train), 'some_train_path.pkl'])
        generate_benchmark_data.__main__([str(num_test), 'some_train_path.pkl'])
        train, test = [result for (result, _), _ in dump_.call_args_list]
        num_features = test['xs'].shape[1]
        num_params = test['theta'].shape[1]
        load_.side_effect = [train, test]

        # Run inference.
        run_inference.__main__([
            '--seed=0',
            algorithm,
            'train.pkl',
            'test.pkl',
            str(num_samples),
            'output.pkl',
        ])
        open_.assert_called_with('output.pkl', 'wb')
        assert open_.call_count == 5
        assert dump_.call_count == 3

    # Get the results.
    (result, _), _ = dump_.call_args
    assert result['posterior_samples'].shape == (num_test, num_samples, num_params)

    if algorithm == 'nunes':
        info = result['info']
        assert info['best_loss'].shape == (num_test,)
        assert info['best_mask'].shape == (num_test, num_features)
        assert info['masks'].shape == (2 ** num_features - 1, num_features)
        assert info['losses'].shape == (2 ** num_features - 1, num_test)
    elif algorithm not in {'naive'}:
        raise NotImplementedError(algorithm)
