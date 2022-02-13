import argparse
import json
import numpy as np
import pickle
from .. import algorithm, benchmark


def preprocess_candidate_features(data: list[dict]):
    """
    Evaluate simple candidate features.
    """
    return np.asarray([np.concatenate([x.mean(axis=0) for x in xs.values()]) for xs in data])


def preprocess_concatenate(data: list[dict]):
    return np.asarray([np.concatenate([x for x in xs.values()]) for xs in data])


ALGORITHMS = {
    'stan': (
        lambda data: np.asarray([x['gaussian_mixture'] for x in data]),
        lambda *_: benchmark.StanBenchmarkAlgorithm()
    ),
    'naive': (
        preprocess_candidate_features,
        lambda d, p, _: algorithm.NearestNeighborAlgorithm(d, p)
    ),
    'nunes': (
        preprocess_candidate_features,
        lambda d, p, _: algorithm.NunesAlgorithm(d, p)
    ),
    'fearnhead': (
        preprocess_concatenate,
        lambda d, p, kwargs: algorithm.FearnheadAlgorithm(d, p, **kwargs)
    ),
    'fearnhead_preprocessed': (
        preprocess_candidate_features,
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


class ListAlgorithmsAction(argparse.Action):
    """
    Action to list all available algorithms.
    """
    def __call__(self, parser: argparse.ArgumentParser, *args, **kwargs):  # pragma: no cover
        print(' '.join(ALGORITHMS))
        parser.exit()


def __main__(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--list', action=ListAlgorithmsAction, help='list all available algorithms',
                        nargs=0)
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
    with open(args.test, 'rb') as fp:
        test = pickle.load(fp)

    # Get a sampling algorithm and optional preprocessor.
    preprocessor, algorithm_cls = ALGORITHMS[args.algorithm]
    train_features = preprocessor(train['data'])
    test_features = preprocessor(test['data'])

    alg: algorithm.Algorithm = algorithm_cls(train_features, train['params'], args.options)
    posterior_samples, info = alg.sample(test_features, args.num_samples)

    # Verify the shape of the posterior samples and save the result.
    expected_shape = (len(test['params']), args.num_samples, alg.num_params)
    assert posterior_samples.shape == expected_shape, 'expected posterior sample shape ' \
        f'{expected_shape} but got {posterior_samples.shape}'

    with open(args.output, 'wb') as fp:
        pickle.dump({
            'args': vars(args),
            'params': test['params'],  # Copy over the true parameter values for easy comparison.
            'posterior_samples': posterior_samples,
            'info': info,
        }, fp)


if __name__ == '__main__':
    __main__()
