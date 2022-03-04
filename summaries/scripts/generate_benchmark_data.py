import argparse
import os
import pickle
import torch as th
from tqdm import tqdm
from .. import benchmark


def __main__(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, help='seed for the random number generator')
    parser.add_argument('--num_observations', type=int, help='number of observations per sample',
                        default=benchmark.NUM_OBSERVATIONS)
    parser.add_argument('--num_noise_features', type=int, help='number of noise features per '
                        'sample', default=benchmark.NUM_NOISE_FEATURES)
    parser.add_argument('num_samples', type=int, help='number of samples to generate')
    parser.add_argument('output', help='output file path')
    args = parser.parse_args(args)

    if args.seed is not None:
        th.manual_seed(args.seed)

    # Disable parameter validation for speedier sampling.
    th.distributions.Distribution.set_default_validate_args(False)

    result = {'args': vars(args)}
    for _ in tqdm(range(args.num_samples)):
        sample = benchmark.sample(num_observations=args.num_observations,
                                  num_noise_features=args.num_noise_features)
        result.setdefault('samples', []).append(sample)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'wb') as fp:
        pickle.dump(result, fp)


if __name__ == '__main__':
    __main__()
