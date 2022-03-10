import argparse
import json
import logging
import numpy as np
import pickle
from .. import algorithm, benchmark, nn, util


def preprocess_candidate_features(samples: dict[str, np.ndarray]):
    """
    Evaluate simple candidate features.
    """
    features = [(samples['x'][..., 0] ** k).mean(axis=-1, keepdims=True) for k in [2, 4, 6, 8]]
    features.append(samples['noise'].mean(axis=-1, keepdims=True))
    return np.hstack(features)


ALGORITHMS = {
    'stan': (
        lambda samples: samples['x'],
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
        preprocess_candidate_features,
        lambda d, p, kwargs: algorithm.FearnheadAlgorithm(d, p, **kwargs)
    ),
    'mdn_compressor': (
        lambda samples: samples['x'],
        lambda d, p, kwargs: nn.NeuralCompressorNearestNeighborAlgorithm(d, p, kwargs['path'])
    ),
    'mdn': (
        lambda samples: samples['x'],
        lambda d, p, kwargs: nn.NeuralAlgorithm(kwargs['path'])
    )
}


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
    parser.add_argument('--cls_options', help='JSON options for the constructor', type=json.loads,
                        default={})
    parser.add_argument('--sample_options', help='JSON options for sampling', type=json.loads,
                        default={})
    parser.add_argument('algorithm', help='algorithm to run', choices=ALGORITHMS)
    parser.add_argument('train', help='training data path')
    parser.add_argument('test', help='test data path')
    parser.add_argument('num_samples', type=int, help='number of samples to generate')
    parser.add_argument('output', help='output file path')
    args = parser.parse_args(args)

    if args.seed is not None:
        np.random.seed(args.seed)

    # Load the training and test data.
    samples_by_split = {}
    for key, path in [('train', args.train), ('test', args.test)]:
        with open(path, 'rb') as fp:
            data = pickle.load(fp)
        samples_by_split[key] = data['samples']

    # Get a sampling algorithm and optional preprocessor.
    preprocessor, algorithm_cls = ALGORITHMS[args.algorithm]
    features_by_split = {key: preprocessor(value) for key, value in samples_by_split.items()}

    # Disable logging for cmdstanpy.
    if args.algorithm == 'stan':
        logger = logging.getLogger('cmdstanpy')
        logger.setLevel(logging.WARNING)

    alg: algorithm.Algorithm = algorithm_cls(
        features_by_split['train'], samples_by_split['train']['theta'], args.cls_options)
    posterior_samples, info = alg.sample(features_by_split['test'], args.num_samples,
                                         **args.sample_options)

    # Verify the shape of the posterior samples and save the result.
    try:
        expected_shape = (len(samples_by_split['test']['theta']), args.num_samples, alg.num_params)
        assert posterior_samples.shape == expected_shape, 'expected posterior sample shape ' \
            f'{expected_shape} but got {posterior_samples.shape}'
    except NotImplementedError:
        pass

    with util.sopen(args.output, 'wb') as fp:
        pickle.dump({
            'args': vars(args),
            'theta': samples_by_split['test']['theta'],
            'posterior_samples': posterior_samples,
            'info': info,
        }, fp)


if __name__ == '__main__':
    __main__()
