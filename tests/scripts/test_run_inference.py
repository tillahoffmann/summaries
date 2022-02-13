import os
import pickle
import pytest
from summaries.scripts import generate_benchmark_data, run_inference
import tempfile


@pytest.mark.parametrize('algorithm', run_inference.ALGORITHMS)
def test_run_inference(algorithm: str):
    num_train = 20
    num_test = 7
    num_samples = 13
    num_params = 2
    num_features = 4

    assert num_train >= num_samples, 'cannot take more samples than there are training points'

    with tempfile.TemporaryDirectory() as directory:
        train_path = os.path.join(directory, 'train.pkl')
        generate_benchmark_data.__main__([str(num_train), train_path])
        test_path = os.path.join(directory, 'test.pkl')
        generate_benchmark_data.__main__([str(num_test), test_path])

        # Run inference.
        output_path = os.path.join(directory, 'output.pkl')
        run_inference.__main__([
            '--seed=0',
            algorithm,
            train_path,
            test_path,
            str(num_samples),
            output_path,
        ])
        with open(output_path, 'rb') as fp:
            result = pickle.load(fp)

    assert result['posterior_samples'].shape == (num_test, num_samples, num_params)

    # Check auxiliary information.
    info = result['info']
    if algorithm == 'nunes':
        assert info['best_loss'].shape == (num_test,)
        assert info['best_mask'].shape == (num_test, num_features)
        assert info['masks'].shape == (2 ** num_features - 1, num_features)
        assert info['losses'].shape == (2 ** num_features - 1, num_test)
    elif algorithm == 'naive' or algorithm.startswith('fearnhead'):
        assert info['distances'].shape == (num_test, num_samples)
        assert info['indices'].shape == (num_test, num_samples)
        assert not algorithm.startswith('fearnhead') \
            or info['predictors'].shape == (num_test, num_params)
    elif algorithm == 'stan':
        pass
    else:
        raise NotImplementedError(algorithm)
