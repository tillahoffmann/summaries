import json
import os
import pickle
import pytest
from summaries import benchmark, nn
from summaries.scripts import generate_benchmark_data, run_inference
import tempfile
import torch as th


@pytest.mark.parametrize('algorithm', run_inference.ALGORITHMS)
def test_run_inference(algorithm: str):
    num_train = 20
    num_test = 7
    num_samples = 13
    num_params = 1
    num_features = 2 + 3  # Three moments for the data and two noise features.
    cls_options = None

    assert num_train >= num_samples, 'cannot take more samples than there are training points'

    with tempfile.TemporaryDirectory() as directory:
        train_path = os.path.join(directory, 'train.pkl')
        generate_benchmark_data.__main__([str(num_train), train_path])
        test_path = os.path.join(directory, 'test.pkl')
        generate_benchmark_data.__main__([str(num_test), test_path])

        if algorithm == 'mdn_compressor':
            compressor = nn.DenseCompressor([1, 8, 3], th.nn.Tanh())
            compressor_path = os.path.join(directory, 'compressor.pt')
            th.save(compressor, compressor_path)
            cls_options = {'path': compressor_path}
        elif algorithm == 'mdn':
            mdn = benchmark.MDNBenchmarkAlgorithm(3, 1)
            mdn_path = os.path.join(directory, 'mdn.pt')
            th.save(mdn, mdn_path)
            cls_options = {'path': mdn_path}

        # Run inference.
        output_path = os.path.join(directory, 'output.pkl')
        args = [
            '--seed=0',
            algorithm,
            train_path,
            test_path,
            str(num_samples),
            output_path,
        ]
        if cls_options:
            args.append(f'--cls_options={json.dumps(cls_options)}')
        run_inference.__main__(args)
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
    elif algorithm in {'naive', 'mdn_compressor'} or algorithm.startswith('fearnhead'):
        assert info['distances'].shape == (num_test, num_samples)
        assert info['indices'].shape == (num_test, num_samples)
        assert not algorithm.startswith('fearnhead') \
            or info['compressed_data'].shape == (num_test, num_params)
    elif algorithm in {'stan', 'mdn'}:
        pass
    else:
        raise NotImplementedError(algorithm)
