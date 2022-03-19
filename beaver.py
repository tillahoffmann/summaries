import beaver_build as bb


def generate_benchmark_data(num_observations):
    """
    Utility function for generating datasets of different sizes with a given number of observations.
    """
    configs = [
        ('train', 1e6, 0),
        ('validation', 1e4, 1),
        ('test', 1e3, 2),
        ('debug', 1e2, 3),
    ]
    for split, size, seed in configs:
        args = [
            'python', '-m', 'summaries.scripts.generate_benchmark_data', int(size), '$@',
            f'--num_observations={num_observations}'
        ]
        bb.Subprocess(f'{split}.pkl', None, args, env={'SEED': seed, 'LOGLEVEL': 'info'})


# Generate synthetic data one small dataset and one large to study the effect of variable numbers of
# observation.
with bb.group_artifacts('workspace', 'benchmark'):
    with bb.group_artifacts('small', 'data'):
        generate_benchmark_data(10)
    with bb.group_artifacts('large', 'data'):
        generate_benchmark_data(100)


# Download data for the benchmark problem, convert to CSV as an intermediate, and then to the same
# pickle format as the benchmark data.
conversion_script = bb.File('summaries/scripts/coal_rda2csv.r')
with bb.group_artifacts('workspace', 'coal', 'data') as (*_, group):
    configs = [
        (
            'coaloracle',
            'https://web.archive.org/web/0if_/https://people.bath.ac.uk/man54/computerstuff/'
            'otherfiles/ABC/coaloracle.rda',
            'a24b2de5',
            {'test': 1_000, 'validation': 10_000, 'train': 989_000},
        ),
        (
            'coal',
            'https://github.com/dennisprangle/abctools/raw/'
            '8c4e440389933722f8288b49bc88c6a38057f511/data/coal.rda',
            'ab2e69b1',
            {'coal': 100_000},
        ),
        (
            'coalobs',
            'https://github.com/dennisprangle/abctools/raw/'
            '8c4e440389933722f8288b49bc88c6a38057f511/data/coalobs.rda',
            '1d48f06c',
            {'obs': 100},
        ),
    ]
    for name, url, digest, splits in configs:
        # Download and convert the data.
        rda, = bb.Download(bb.File(f'{name}.rda', expected_digest=digest), url)
        csv, = bb.Subprocess(f'{name}.csv', [conversion_script, rda],
                             ['Rscript', '--vanilla', conversion_script, rda, name, '$@'])

        # Split the data into smaller parts for training, validation, and test.
        filenames = [f'{split}.pkl' for split in splits]
        splits = [f'{split}.pkl={size}' for split, size in splits.items()]
        cmd = ["$!", "-m", "summaries.scripts.preprocess_coal", csv, group.name, *splits]
        bb.Subprocess(filenames, csv, cmd, env={'SEED': 0})


def train_nn(problem, architecture, num_features, num_components):
    """Utility function to train compressors."""
    outputs = [f"{architecture}.pt"]
    with bb.group_artifacts('data'):
        inputs = [bb.File('train.pkl'), bb.File('validation.pkl')]

    args = [
        '$!', '-m', 'summaries.scripts.train_nn', problem, architecture, *inputs, '$@',
        f'--num_features={num_features}',
    ]
    # Also save the mixture density network rather than just the compressor.
    if architecture == 'mdn_compressor':
        mdn = bb.File("mdn.pt")
        outputs.append(mdn)
        args.extend([f'--mdn_output={mdn}', f'--num_components={num_components}'])
    else:
        assert num_components is None
    bb.Subprocess(outputs, inputs, args, env={'SEED': 1, 'LOGLEVEL': 'INFO'})


for size in ['small', 'large']:
    with bb.group_artifacts('workspace', 'benchmark', size):
        train_nn('benchmark', 'mdn_compressor', 1, 2)
        train_nn('benchmark', 'regressor', 1, None)

with bb.group_artifacts('workspace', 'coal'):
    train_nn('coal', 'mdn_compressor', 2, 10)
    train_nn('coal', 'regressor', 2, None)


def sample(problem, method, num_samples, output=None, path=None):
    """
    Utility function for drawing posterior samples.
    """
    with bb.group_artifacts('data'):
        inputs = [bb.File("train.pkl"), bb.File("test.pkl")]

    algorithm = method
    flags = {}
    if method in {'mdn_compressor', 'mdn', 'regressor'}:
        path = path or bb.File(f'{method}.pt')
        inputs.append(path)
        flags['cls_options'] = '{{"path": "%s"}}' % path.name
        if method != 'mdn':
            algorithm = 'neural_compressor'
    elif method == 'stan':
        flags['sample_options'] = '{{"keep_fits": true, "seed": 0, "adapt_delta": 0.99}}'
    args = ['$!', '-m', 'summaries.scripts.run_inference', problem, algorithm, *inputs[:2],
            num_samples, '$@'] + [f'--{key}={value}' for key, value in flags.items()]
    bb.Subprocess(output or f"samples/{method}.pkl", inputs, args)


# Run on the benchmark problem.
methods = ['naive', 'fearnhead', 'nunes', 'regressor', 'mdn_compressor', 'mdn']
for method in methods + ['stan']:
    with bb.group_artifacts('workspace', 'benchmark', 'small'):
        sample('benchmark', method, 5000)

# Just run the "ground truth" and compressor trained on the large dataset for the large dataset.
for method in ['stan', 'mdn_compressor']:
    with bb.group_artifacts('workspace', 'benchmark', 'large'):
        sample('benchmark', method, 5000)

# Add on mdn compression samples for the statistics we learned with the small dataset but apply them
# to the large dataset. This allows us to study how good the statistics are at generalising to
# datasets of different sizes.
compressor = bb.File('workspace/benchmark/small/mdn_compressor.pt')
with bb.group_artifacts('workspace', 'benchmark', 'large'):
    output = "samples/mdn_compressor_small.pkl"
    sample('benchmark', 'mdn_compressor', 5000, output, compressor)

# Apply to coalescent dataset.
for method in methods:
    with bb.group_artifacts('workspace', 'coal'):
        # 4945 samples ensures that we sample the same fraction of the reference table: 0.5%.
        sample('coal', method, 4945)
