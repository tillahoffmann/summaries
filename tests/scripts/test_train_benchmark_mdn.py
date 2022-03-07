import os
from summaries.scripts import generate_benchmark_data, train_benchmark_mdn
import tempfile
import torch as th


def test_train_benchmark_mdn():
    num_components = 7
    num_features = 3

    with tempfile.TemporaryDirectory() as tmp:
        # Generate some data for training.
        data_path = os.path.join(tmp, 'train.pkl')
        generate_benchmark_data.__main__(['10', data_path])

        # Train the model.
        mdn_path = os.path.join(tmp, 'mdn.pt')
        compressor_path = os.path.join(tmp, 'compressor.pt')
        args = [data_path, data_path, mdn_path, compressor_path,
                f'--num_components={num_components}', f'--num_features={num_features}']
        train_benchmark_mdn.__main__(args)

        # Load the models and validate the output.
        batch_size = 17
        x = th.randn(batch_size, 10, 1)
        mdn = th.load(mdn_path)
        dist: th.distributions.MixtureSameFamily = mdn(x)
        dist._mixture_distribution.event_shape == (num_components,)
        dist.batch_shape == (batch_size,)

        # Load the compressor and validate the output.
        compressor = th.load(compressor_path)
        y = compressor(x)
        assert y.shape == (batch_size, num_features)
