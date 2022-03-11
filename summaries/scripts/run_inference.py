import argparse
import cmdstanpy
import json
import logging
import numpy as np
import pickle
from .. import algorithm, benchmark, nn, util


def preprocess_candidate_features(samples: dict[str, np.ndarray]):
    """
    Evaluate simple candidate features.
    """
    return np.hstack([
        np.mean(samples['x'] ** [2, 4, 6, 8], axis=-2),
        np.mean(samples['noise'], axis=-2)
    ])


def concatenate_features(samples: dict[str, np.ndarray]):
    """
    Concatenate features into a single feature vector.
    """
    return np.concatenate([samples['x'], samples['noise']], axis=-1)


ALGORITHMS_BY_MODEL = {
    'benchmark': {
        'stan': (
            lambda samples: samples['x'],
            lambda *_: benchmark.StanBenchmarkAlgorithm(),
        ),
        'naive': (
            preprocess_candidate_features,
            lambda d, p, _: algorithm.NearestNeighborAlgorithm(d, p),
        ),
        'nunes': (
            preprocess_candidate_features,
            lambda d, p, _: algorithm.NunesAlgorithm(d, p),
        ),
        'fearnhead': (
            preprocess_candidate_features,
            lambda d, p, kwargs: algorithm.FearnheadAlgorithm(d, p, **kwargs),
        ),
        'mdn_compressor': (
            concatenate_features,
            lambda d, p, kwargs: nn.NeuralCompressorNearestNeighborAlgorithm(d, p, kwargs['path']),
        ),
        'mdn': (
            concatenate_features,
            lambda d, p, kwargs: nn.NeuralAlgorithm(kwargs['path']),
        )
    },
    'coal': {
        'naive': (
            lambda samples: samples['x'],
            lambda d, p, kwargs: algorithm.NearestNeighborAlgorithm(d, p),
        ),
        'fearnhead': (
            lambda samples: samples['x'],
            lambda d, p, kwargs: algorithm.FearnheadAlgorithm(d, p),
        ),
        'nunes': (
            lambda samples: samples['x'],
            lambda d, p, kwargs: algorithm.NunesAlgorithm(d, p),
        )
    }
}
ALGORITHMS = set(algo for algos in ALGORITHMS_BY_MODEL.values() for algo in algos)


def __main__(args=None):
    util.setup_script()
    cmdstanpy.utils.get_logger().setLevel(logging.WARNING)

    parser = argparse.ArgumentParser()
    parser.add_argument('--cls_options', help='JSON options for the constructor', type=json.loads,
                        default={})
    parser.add_argument('--sample_options', help='JSON options for sampling', type=json.loads,
                        default={})
    parser.add_argument('model', help='model for inference; needed for preprocessing',
                        choices=['benchmark', 'coal'])
    parser.add_argument('algorithm', help='algorithm to run', choices=ALGORITHMS)
    parser.add_argument('train', help='training data path')
    parser.add_argument('test', help='test data path')
    parser.add_argument('num_samples', type=int, help='number of samples to generate')
    parser.add_argument('output', help='output file path')
    args = parser.parse_args(args)

    # Load the training and test data.
    samples_by_split = {}
    for key, path in [('train', args.train), ('test', args.test)]:
        with open(path, 'rb') as fp:
            data = pickle.load(fp)
        samples_by_split[key] = data['samples']

    # Get a sampling algorithm and optional preprocessor.
    preprocessor, algorithm_cls = ALGORITHMS_BY_MODEL[args.model][args.algorithm]
    features_by_split = {key: preprocessor(value) for key, value in samples_by_split.items()}

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
