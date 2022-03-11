import argparse
import cmdstanpy
import json
import logging
import numpy as np
import pickle
from .. import algorithm, benchmark, nn, util


def preprocess_candidate_features(x: np.ndarray) -> np.ndarray:
    """
    Evaluate simple candidate features.
    """
    return np.hstack([
        np.mean(x[..., :1] ** [2, 4, 6, 8], axis=-2),
        np.mean(x[..., 1:], axis=-2)
    ])


ALGORITHMS_BY_MODEL = {
    'benchmark': {
        'stan': (None, lambda *_: benchmark.StanBenchmarkAlgorithm()),
        'naive': (preprocess_candidate_features, algorithm.NearestNeighborAlgorithm),
        'nunes': (preprocess_candidate_features, algorithm.NunesAlgorithm),
        'fearnhead': (preprocess_candidate_features, algorithm.FearnheadAlgorithm),
        'mdn_compressor': (None, nn.NeuralCompressorNearestNeighborAlgorithm),
        'mdn': (None, lambda *_, **kwargs: nn.NeuralAlgorithm(**kwargs)),
    },
    'coal': {
        'naive': (None, algorithm.NearestNeighborAlgorithm),
        'fearnhead': (None, algorithm.FearnheadAlgorithm),
        'nunes': (None, algorithm.NunesAlgorithm),
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
    preprocessor = preprocessor or (lambda x: x)
    features_by_split = {key: preprocessor(value['x']) for key, value in samples_by_split.items()}

    alg: algorithm.Algorithm = algorithm_cls(
        features_by_split['train'], samples_by_split['train']['theta'], **args.cls_options)
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
