import argparse
import numpy as np
import pickle
import torch as th
from tqdm import tqdm
from .. import benchmark
from .. import util


def __main__(args=None):
    util.setup_script()

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_observations', type=int, help='number of observations per sample',
                        default=benchmark.NUM_OBSERVATIONS)
    parser.add_argument('--num_noise_features', type=int, help='number of noise features per '
                        'sample', default=benchmark.NUM_NOISE_FEATURES)
    parser.add_argument('--batch_size', help='batch size for generating samples', type=int,
                        default=100)
    parser.add_argument('num_samples', type=int, help='number of samples to generate')
    parser.add_argument('output', help='output file path')
    args = parser.parse_args(args)

    # Disable parameter validation for speedier sampling.
    th.distributions.Distribution.set_default_validate_args(False)

    samples = {}
    times = []
    num_samples = 0
    with tqdm(total=args.num_samples) as progress:
        while num_samples < args.num_samples:
            batch_size = min(args.batch_size, args.num_samples - num_samples)
            sample = benchmark.sample(num_observations=args.num_observations, size=batch_size,
                                      num_noise_features=args.num_noise_features)
            for key, value in sample.items():
                if isinstance(value, th.Tensor):
                    value = value.numpy()
                samples.setdefault(key, []).append(value)
            num_samples += batch_size
            progress.update(batch_size)
    samples = {key: np.concatenate(value, axis=0) for key, value in samples.items()}
    result = {
        'args': vars(args),
        'samples': samples,
        'times': np.asarray(times),
    }

    with util.sopen(args.output, 'wb') as fp:
        pickle.dump(result, fp)


if __name__ == '__main__':
    __main__()
