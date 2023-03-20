from doit.action import CmdAction
from doit.tools import run_once
import functools as ft
import os
import pathlib
import sys
import typing


# Standard environment variables to avoid interaction between different processes.
ENV = os.environ | {
    "NUMEXPR_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
}
cmd_action = ft.partial(CmdAction, shell=False)
default_task = {
    "io": {
        "capture": False,
    }
}

# Interpreter for launching subtasks.
PYTHON = sys.executable
# Arguments for starting a subprocess that halts on exceptions.
DEBUG_ARGS = [PYTHON, "-m", "pdb", "-c", "continue"]
# Root directory for generating results.
ROOT = pathlib.Path("workspace")


def task_benchmark_data():
    """
    Generate data for the benchmark problem.
    """
    configs = {
        "train": (1_000_000, 0),
        "validation": (10_000, 1),
        "test": (1_000, 2),
        "debug": (100, 3),
    }
    observation_sizes = {
        "small": 10,
        "large": 100,
    }
    for observation_key, num_observations in observation_sizes.items():
        for split, (sample_size, seed) in configs.items():
            basename = f"benchmark/{observation_key}/data"
            target = ROOT / basename / f"{split}.pkl"
            args = [
                PYTHON, "-m", 'summaries.scripts.generate_benchmark_data', str(sample_size), target,
                f'--num_observations={num_observations}'
            ]
            env = {
                "SEED": str(num_observations + seed)
            }
            action = cmd_action(args, env=ENV | env)
            yield default_task | {
                "basename": basename,
                "name": split,
                "actions": [action],
                "targets": [target],
                "uptodate": [run_once],
            }


def task_coal_data():
    """
    Download data for the coalescent problem and unpack them.
    """
    # Download the data.
    url = "https://github.com/tillahoffmann/coaloracle/releases/download/0.2/csv.zip"
    basename = "coal/data"
    name = "csv.zip"
    directory = ROOT / basename
    zip_target = directory / name
    yield default_task | {
        "basename": basename,
        "name": name,
        "actions": [
            ["mkdir", "-p", directory],
            ["curl", "-L", "-o", zip_target, url]
        ],
        "targets": [zip_target],
        "uptodate": [run_once],
    }

    # Define splits for each of the datasets we download.
    splits_by_name = {
        "coal": {
            "coal": 100_000,
        },
        "coalobs": {
            "obs": 100,
        },
        "coaloracle": {
            "test": 1_000,
            "validation": 10_000,
            "train": 989_000,
        }
    }

    # Unpack the data and preprocess.
    for name, splits in splits_by_name.items():
        csv_target = ROOT / basename / f"{name}.csv"
        yield default_task | {
            "basename": basename,
            "name": name,
            "actions": [["unzip", "-o", "-j", "-d", directory, zip_target, f"csv/{name}.csv"]],
            "file_dep": [zip_target],
            "targets": [csv_target],
        }

        targets = [ROOT / basename / f'{split}.pkl' for split in splits]
        splits = [f'{split}.pkl={size}' for split, size in splits.items()]
        args = [PYTHON, "-m", "summaries.scripts.preprocess_coal", csv_target, directory, *splits]

        yield default_task | {
            "basename": basename,
            "name": f"{name}-splits",
            "targets": targets,
            "actions": [cmd_action(args, env=ENV | {"SEED": "0"})],
            "file_dep": [csv_target],
        }


def train_nn(problem: typing.Literal["benchmark", "coal"],
             architecture: typing.Literal["mdn_compressor", "regressor"], num_features: int,
             num_components: int):
    """
    Utility function to train compressors.

    Args:
        problem: Inference problem to train a model for.
        architecture: Architecture of the model to train.
        num_features: Number of features to learn.
        num_components: Number of mixture components (only for `mdn_compressor` architecture).
    """
    directory = ROOT / problem
    inputs = [directory / "data" / f"{split}.pkl" for split in ["train", "validation"]]
    compressor_target = directory / f"{architecture}.pt"
    targets = [compressor_target]
    # We pick the first part of the `problem` argument as the problem to pass to the training script
    # so we can train the benchmark for both a small and a large sample size.
    args = [
        PYTHON, "-m", "summaries.scripts.train_nn", problem.split('/')[0], architecture,
        *inputs, compressor_target, f"--num_features={num_features}"
    ]
    if architecture == "mdn_compressor":
        mdn_target = directory / "mdn.pt"
        args.extend([f'--mdn_output={mdn_target}', f'--num_components={num_components}'])
    elif num_components is not None:
        raise ValueError("components are only applicable to mixture density networks")

    return default_task | {
        "basename": problem,
        "name": architecture,
        "actions": [cmd_action(args, env=ENV | {"SEED": "1", "LOGLEVEL": "INFO"})],
        "targets": targets,
        "file_dep": inputs,
    }


def task_train_nn():
    """
    Train compressor neural networks.
    """
    for size in ["small", "large"]:
        problem = f"benchmark/{size}"
        yield train_nn(problem, "mdn_compressor", 1, 2)
        yield train_nn(problem, "regressor", 1, None)

    yield train_nn("coal", "mdn_compressor", 2, 10)
    yield train_nn("coal", "regressor", 2, None)


def sample(problem: typing.Literal["benchmark", "coal"], method: str, num_samples: int, *,
           target: str = None, model_path: str = None, name: str = None):
    """
    Utility function to draw posterior samples.

    Args:
        problem: Inference problem to draw samples for.
        method: Method for drawing posterior samples.
        num_samples: Number of samples to draw.
        target: Explicit output path (constructed based on `problem` and `method` if not given).
        model_path: Explicit model input path (constructed based on `problem` and `method` if not
            given).
    """
    basename = f"{problem}/samples"
    directory = ROOT / problem
    inputs = [directory / "data" / f"{split}.pkl" for split in ["train", "test"]]
    target = target or directory / f"samples/{method}.pkl"

    algorithm = method
    flags = {}
    if method in {"mdn_compressor", "mdn", "regressor"}:
        model_path = model_path or directory / f"{method}.pt"
        inputs.append(model_path)
        flags["cls_options"] = '{"path": "%s"}' % model_path
        if method != 'mdn':
            algorithm = 'neural_compressor'
    elif model_path is not None:
        raise ValueError(f"explicit model path {model_path} cannot be used with method {method}")
    elif method == 'stan':
        flags['sample_options'] = '{"keep_fits": false, "seed": 0, "adapt_delta": 0.99}'
    args = [PYTHON, '-m', 'summaries.scripts.run_inference', problem.split("/")[0], algorithm,
            *inputs[:2], str(num_samples), target] \
        + [f'--{key}={value}' for key, value in flags.items()]

    return default_task | {
        "basename": basename,
        "name": name or method,
        "actions": [cmd_action(args, env=ENV)],
        "file_dep": inputs,
        "targets": [target],
    }


def task_sample_posterior():
    """
    Draw posterior samples.
    """
    # Run on the benchmark problem.
    methods = ['naive', 'fearnhead', 'nunes', 'regressor', 'mdn_compressor', 'mdn']
    for method in methods + ['stan']:
        yield sample("benchmark/small", method, 5000)

    # Just run the "ground truth" and compressor trained on the large dataset for the large dataset.
    for method in ["stan", "mdn_compressor"]:
        yield sample("benchmark/large", method, 5000)

    # The Fearnhead dataset suffers from additional variability because the regression coefficients
    # are wholly determined by noise: there is no (linear) signal. This script captures additional
    # variability by repeated runs. This is neither necessary nor computationally feasible for the
    # other methods.
    basename = "benchmark/small/fearnhead_random_entropies"
    target = ROOT / f"{basename}.pkl"
    args = [PYTHON, "-m", "summaries.scripts.estimate_benchmark_entropy", "--batch_size=100",
            "1000000", "100", target]
    yield default_task | {
        "targets": [target],
        "basename": basename,
        "actions": [cmd_action(args, env=ENV | {"SEED": "0"})],
        "uptodate": [run_once],
    }

    # Add on MDN compression samples for the statistics we learned with the small dataset but apply
    # them to the large dataset. This allows us to study how good the statistics are at generalizing
    # to datasets of different sizes.
    target = ROOT / "benchmark/large/samples/mdn_compressor_small.pkl"
    compressor = ROOT / "benchmark/small/mdn_compressor.pt"
    yield sample("benchmark/large", "mdn_compressor", 5000, target=target, model_path=compressor,
                 name="mdn_compressor_small")

    # Apply to coalescent dataset.
    for method in methods:
        # 4945 samples ensures that we sample the same fraction of the reference table: 0.5%.
        yield sample('coal', method, 4945)


def task_figures():
    """
    Generate figures.
    """
    basename = "figures"
    directory = ROOT / basename
    args = [PYTHON, "-m", "nbconvert", "--execute", "--to=html", f"--output-dir={directory}",
            "figures.ipynb"]
    yield {
        "basename": basename,
        "actions": [args],
        "targets": [directory / "figures.ipynb"],
        "file_dep": ["figures.ipynb"],
    }
