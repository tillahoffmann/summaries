import numpy as np
import os
import pickle
import pytest
from summaries.scripts import train_nn
import tempfile
import torch as th


@pytest.mark.parametrize('model', ['coal', 'benchmark'])
@pytest.mark.parametrize('architecture', ['mdn', 'regressor'])
def test_train_nn(model: str, architecture: str):
    num_components = 7
    batch_size = 13

    if model == 'coal':
        num_dims = 7
        num_params = 2
        data_shape = (batch_size, num_dims)
    elif model == 'benchmark':
        num_dims = 3
        num_params = 1
        data_shape = (batch_size, 11, num_dims)
    else:
        raise NotImplementedError(model)
    param_shape = (batch_size, num_params)

    if architecture == 'mdn':
        num_features = 2 * num_params
    elif architecture == 'regressor':
        num_features = num_params
    else:
        raise NotImplementedError(architecture)

    with tempfile.TemporaryDirectory() as tmp:
        # Generate some data for training.
        data = {
            'samples': {
                'x': np.random.normal(0, 1, data_shape),
                # We sample from the support of the coalescent model posterior or the tests fail.
                'theta': np.random.uniform(0, 10, param_shape),
            }
        }
        data_path = os.path.join(tmp, 'train.pkl')
        with open(data_path, 'wb') as fp:
            pickle.dump(data, fp)

        # Train the model.
        compressor_path = os.path.join(tmp, 'compressor.pt')
        args = [
            model, architecture, data_path, data_path, compressor_path,
            f'--num_components={num_components}', f'--num_features={num_features}',
        ]
        if architecture == 'mdn':
            mdn_path = os.path.join(tmp, 'mdn.pt')
            args.append(f'--mdn_output={mdn_path}')
        train_nn.__main__(args)

        # Load the compressor and validate the output.
        compressor = th.load(compressor_path)
        x = th.as_tensor(data['samples']['x'])
        y = compressor(x)
        assert y.shape == (batch_size, num_features)

        # Validate the mixture distribution if that' the architecture we're using.
        if architecture == 'mdn':
            mdn = th.load(mdn_path)
            dist: th.distributions.MixtureSameFamily = mdn(x)
            assert dist.batch_shape == (batch_size,)
            assert dist.event_shape == (num_params,)
