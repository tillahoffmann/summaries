import json
import numpy as np
import os
import pickle
import pytest
from summaries import benchmark, coal, nn
from summaries.scripts import run_inference
import tempfile
import torch as th


@pytest.mark.parametrize('model', run_inference.ALGORITHMS_BY_MODEL)
@pytest.mark.parametrize('algorithm', run_inference.ALGORITHMS)
def test_run_inference(model: str, algorithm: str):
    num_train = 20
    num_test = 7
    num_samples = 13
    assert num_train >= num_samples, 'cannot take more samples than there are training points'

    if model == 'coal':
        num_dims = num_features = 7
        num_params = 2
        obs_shape = ()
    elif model == 'benchmark':
        num_dims = 3
        num_features = 4 + benchmark.NUM_NOISE_FEATURES
        num_params = 1
        obs_shape = (10,)
    else:
        raise NotImplementedError(model)

    cls_options = None

    with tempfile.TemporaryDirectory() as directory:
        # Generate data to test on.
        train_path = os.path.join(directory, 'train.pkl')
        test_path = os.path.join(directory, 'test.pkl')
        for path, size in [(train_path, num_train), (test_path, num_test)]:
            data = {
                'samples': {
                    'x': np.random.normal(0, 1, (size, *obs_shape, num_dims)),
                    'theta': np.random.uniform(0, 10, (size, num_params)),
                }
            }
            with open(path, 'wb') as fp:
                pickle.dump(data, fp)

        # Run algorithm-specific preparation.
        if algorithm == 'stan' and model == 'coal':
            pytest.skip('likelihood is not available for coalescent model')

        if algorithm in {'mdn', 'neural_compressor'}:
            nn_path = os.path.join(directory, 'nn.pt')

            if model == 'benchmark':
                module = nn.DenseCompressor([num_dims, num_params + 1], th.nn.Tanh())
            elif model == 'coal':
                module = nn.DenseStack([num_dims, num_params + 1], th.nn.Tanh())
            else:
                raise NotImplementedError(model)

            if algorithm == 'mdn':
                if model == 'benchmark':
                    module = benchmark.MixtureDensityNetwork(module, [num_params + 1, 7],
                                                             th.nn.Tanh())
                elif model == 'coal':
                    module = coal.MixtureDensityNetwork(module, [num_params + 1, 7], th.nn.Tanh())
                else:
                    raise NotImplementedError(model)

            th.save(module, nn_path)
            cls_options = {'path': nn_path}

        # Run inference.
        output_path = os.path.join(directory, 'output.pkl')
        args = [
            model,
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
    elif algorithm in {'naive', 'neural_compressor', 'fearnhead'}:
        assert info['distances'].shape == (num_test, num_samples)
        assert info['indices'].shape == (num_test, num_samples)
        assert not algorithm.startswith('fearnhead') \
            or info['compressed_data'].shape == (num_test, num_params)
    elif algorithm in {'stan', 'mdn'}:
        pass
    else:
        raise NotImplementedError(algorithm)
