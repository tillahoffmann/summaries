import argparse
import json
import numpy as np
import pickle
from .. import algorithm, benchmark


def evaluate_candidate_features(data):
    """
    Evaluate simple candidate features.
    """
    return data.mean(axis=-1)


ALGORITHMS = {
    'stan': (
        None,
        lambda *_: benchmark.StanBenchmarkAlgorithm()
    ),
    'naive': (
        evaluate_candidate_features,
        lambda d, p, _: algorithm.NearestNeighborAlgorithm(d, p)
    ),
    'nunes': (
        evaluate_candidate_features,
        lambda d, p, _: algorithm.NunesAlgorithm(d, p)
    ),
    'fearnhead': (
        None,
        lambda d, p, kwargs: algorithm.FearnheadAlgorithm(d, p, **kwargs)
    ),
    'fearnhead_preprocessed': (
        None,
        lambda d, p, kwargs: algorithm.FearnheadAlgorithm(d, p, **kwargs)
    ),
}


class _Args(argparse.Namespace):
    seed: int
    options: dict
    algorithm: str
    train: str
    test: str
    num_samples: int
    output: str


def __main__(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, help='seed for the random number generator')
    parser.add_argument('--options', help='JSON options for the sampler', type=json.loads,
                        default={})
    parser.add_argument('algorithm', help='algorithm to run', choices=ALGORITHMS)
    parser.add_argument('train', help='training data path')
    parser.add_argument('test', help='test data path')
    parser.add_argument('num_samples', type=int, help='number of samples to generate')
    parser.add_argument('output', help='output file path')
    args: _Args = parser.parse_args(args)

    if args.seed is not None:
        np.random.seed(args.seed)

    # Load the training and test data.
    with open(args.train, 'rb') as fp:
        train = pickle.load(fp)
        train_data = train['xs']
        train_params = train['theta']
    with open(args.test, 'rb') as fp:
        test = pickle.load(fp)
        test_data = test['xs']
        test_params = test['theta']

    # Get a sampling algorithm and optional preprocessor.
    preprocessor, algorithm_cls = ALGORITHMS[args.algorithm]
    if preprocessor:
        train_data = preprocessor(train_data)
        test_data = preprocessor(test_data)

    alg: algorithm.Algorithm = algorithm_cls(train_data, train_params, args.options)
    posterior_samples, info = alg.sample(test_data, args.num_samples)

    # Verify the shape of the posterior samples and save the result.
    expected_shape = (test['theta'].shape[0], args.num_samples, alg.num_params)
    assert posterior_samples.shape == expected_shape, 'expected posterior sample shape ' \
        f'{expected_shape} but got {posterior_samples.shape}'

    with open(args.output, 'wb') as fp:
        pickle.dump({
            'args': vars(args),
            'theta': test_params,  # Copy over the true parameter values for easy comparison.
            'posterior_samples': posterior_samples,
            'info': info,
        }, fp)


if __name__ == '__main__':
    __main__()
