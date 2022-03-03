import argparse
import numpy as np
import pickle
from tqdm import tqdm
from .. import benchmark


def __main__(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, help='seed for the random number generator')
    parser.add_argument('--num_observations', type=int, help='number of observations per '
                        'likelihood and sample', default=5)
    parser.add_argument('num_samples', type=int, help='number of samples to generate')
    parser.add_argument('output', help='output file path')
    args = parser.parse_args(args)

    if args.seed is not None:
        np.random.seed(args.seed)

    result = {'args': vars(args)}
    for _ in tqdm(range(args.num_samples)):
        theta = np.random.normal(0, 1)
        data = benchmark.sample(benchmark.LIKELIHOODS, theta, args.num_observations)
        result.setdefault('data', []).append(data)
        result.setdefault('params', []).append(theta)
    result['params'] = np.asarray(result['params'])

    with open(args.output, 'wb') as fp:
        pickle.dump(result, fp)


if __name__ == '__main__':
    __main__()
