import argparse
import numpy as np
import pickle
import time
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
    parser.add_argument('num_samples', type=int, help='number of samples to generate')
    parser.add_argument('output', help='output file path')
    args = parser.parse_args(args)

    # Disable parameter validation for speedier sampling.
    th.distributions.Distribution.set_default_validate_args(False)

    samples = {}
    times = []
    for _ in tqdm(range(args.num_samples)):
        start = time.time()
        sample = benchmark.sample(num_observations=args.num_observations,
                                  num_noise_features=args.num_noise_features)
        times.append(time.time() - start)
        for key, value in sample.items():
            if isinstance(value, th.Tensor):
                value = value.numpy()
            samples.setdefault(key, []).append(value)
    samples = {key: np.asarray(value) for key, value in samples.items()}
    result = {
        'args': vars(args),
        'samples': samples,
        'times': np.asarray(times),
    }

    with util.sopen(args.output, 'wb') as fp:
        pickle.dump(result, fp)


if __name__ == '__main__':
    __main__()
