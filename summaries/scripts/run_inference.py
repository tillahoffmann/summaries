import argparse
import json
import numpy as np
import pickle
import summaries


def naive_rejection_sampler(train_data: dict, _: dict) -> summaries.RejectionAlgorithm:
    return summaries.RejectionAlgorithm(train_data['xs'], train_data['theta'],
                                        transform=lambda x: x.mean(axis=-1))


ALGORITHMS = {
    'naive': naive_rejection_sampler,
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
    parser.add_argument('--options', help='JSON options for the sampler', type=json.loads)
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
        train_data = pickle.load(fp)
    with open(args.test, 'rb') as fp:
        test_data = pickle.load(fp)

    # Create a sampler and run it on the entire test set.
    algorithm: summaries.Algorithm = ALGORITHMS[args.algorithm](train_data, args.options)
    posterior_samples = algorithm.sample_posterior(test_data['xs'], args.num_samples)

    # Verify the shape of the posterior samples and save the result.
    expected_shape = (test_data['theta'].shape[0], args.num_samples, algorithm.num_params)
    assert posterior_samples.shape == expected_shape, 'expected posterior sample shape ' \
        f'{expected_shape} but got {posterior_samples.shape}'

    with open(args.output, 'wb') as fp:
        pickle.dump({
            'args': vars(args),
            'theta': test_data['theta'],  # Copy over the true parameter values for easy comparison.
            'posterior_samples': posterior_samples,
        }, fp)


if __name__ == '__main__':
    __main__()
