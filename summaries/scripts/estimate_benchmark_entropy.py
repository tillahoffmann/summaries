import argparse
import numpy as np
import pickle
from tqdm import tqdm
from .. import algorithm, benchmark, util


def __main__(args: list[str] = None) -> int:
    util.setup_script()

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, help="number of test samples per run",
                        default=100)
    parser.add_argument("--num_posterior_samples", type=int, default=5000,
                        help="number of posterior samples to draw for entropy estimation")
    parser.add_argument("num_reference_samples", type=int,
                        help="number of samples in the reference table per run")
    parser.add_argument("num_repeats", help="number of independent runs to capture variability",
                        type=int),
    parser.add_argument("output", help="path for results")
    args = parser.parse_args(args)

    entropies = {}
    coefs = {}
    for _ in tqdm(range(args.num_repeats)):
        # Generate training and test data.
        train_batch = benchmark.sample(size=args.num_reference_samples)
        train_features = benchmark.preprocess_candidate_features(train_batch["x"].numpy())
        test_batch = benchmark.sample(size=args.batch_size)
        test_features = benchmark.preprocess_candidate_features(test_batch["x"].numpy())

        coef = np.random.normal(0, 1, (train_features.shape[1], 1))
        fearnhead = algorithm.FearnheadAlgorithm(train_features, train_batch["theta"])
        algorithms = {
            "fearnhead": fearnhead,
            "random": algorithm.StaticCompressorNearestNeighborAlgorithm(
                train_features, train_batch["theta"], lambda x: x @ coef
            )
        }

        coefs.setdefault("random", []).append(coef)
        coefs.setdefault("fearnhead", []).append(fearnhead.predictor.coef_)

        for method, algo in algorithms.items():
            # Draw samples and estimate entropies.
            samples, _ = algo.sample(test_features, args.num_posterior_samples)
            entropies.setdefault(method, []).append([util.estimate_entropy(x) for x in samples])

    # Cast to numpy and report results informally.
    entropies = {key: np.asarray(value) for key, value in entropies.items()}
    for key, value in entropies.items():
        value = value.mean(axis=1)
        print(f'{key}: {value.mean():.3f} +- {value.std() / np.sqrt(value.size - 1):.3f}')

    # Cast coefficients to numpy and report.
    coefs = {key: np.squeeze(value) for key, value in coefs.items()}
    for key, value in coefs.items():
        print(f"{key} coefficients")
        print("mean", value.mean(axis=0))
        print("std", value.std(axis=0))

    # Save the results.
    with open(args.output, 'wb') as fp:
        pickle.dump({
            'args': vars(args),
            'entropies': entropies,
        }, fp)


if __name__ == '__main__':
    __main__()
