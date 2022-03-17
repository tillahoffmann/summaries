import beaver_build as bb
import itertools as it


# Generate synthetic data.
with bb.group_artifacts('workspace', 'benchmark', 'data'):
    configs = [
        ('train', 1e6, 0),
        ('validation', 1e4, 1),
        ('test', 1e3, 2),
        ('debug', 1e2, 3),
    ]
    for split, size, seed in configs:
        args = ['python', '-m', 'summaries.scripts.generate_benchmark_data', int(size), '$@']
        bb.Subprocess(f'{split}.pkl', None, args, env={'SEED': seed})


# Download data for the benchmark problem, convert to CSV as an intermediate, and then to the same
# pickle format as the benchmark data.
conversion_script = bb.File('summaries/scripts/coal_rda2csv.r')
with bb.group_artifacts('workspace', 'coal', 'data') as (*_, group):
    configs = [
        (
            'coaloracle',
            'https://web.archive.org/web/0if_/https://people.bath.ac.uk/man54/computerstuff/'
            'otherfiles/ABC/coaloracle.rda',
            '7a8a639bd0ec9d017a25e6b7fb4da0b9d3e8cc8c1a24c75b21340aab88a7122f',
            {'test': 1_000, 'validation': 10_000, 'train': 989_000},
        ),
        (
            'coal',
            'https://github.com/dennisprangle/abctools/raw/'
            '8c4e440389933722f8288b49bc88c6a38057f511/data/coal.rda',
            '1b52cd9d166daffdcddac71e4edfc9663d6d06c97728c2cbd6dcaf5ea665d365',
            {'coal': 100_000},
        ),
        (
            'coalobs',
            'https://github.com/dennisprangle/abctools/raw/'
            '8c4e440389933722f8288b49bc88c6a38057f511/data/coalobs.rda',
            '3aa189cd40f1e43e3755cee8debed4f31a5f86bafaa42b6584591eca1b3fba63',
            {'obs': 100},
        ),
    ]
    for name, url, digest, splits in configs:
        rda, = bb.Download(f'{name}.rda', url, digest)

        csv, = bb.Subprocess(f'{name}.csv', [conversion_script, rda],
                             ['Rscript', '--vanilla', '$^', name, '$@'])

        filenames = [f'{split}.pkl' for split in splits]
        splits = [f'{split}.pkl={size}' for split, size in splits.items()]
        cmd = ["$!", "-m", "summaries.scripts.preprocess_coal", "$<", group.name, ' '.join(splits)]
        bb.Subprocess(filenames, csv, cmd, env={'SEED': 0})


# Train the machine learning models for the coalescent and benchmark problems.
for problem, architecture in it.product(['benchmark', 'coal'], ['mdn_compressor', 'regressor']):
    with bb.group_artifacts('workspace', problem):
        outputs = [f"{architecture}.pt"]
        inputs = [bb.File('data/train.pkl'), bb.File('data/validation.pkl')]
        args = ['$!', '-m', 'summaries.scripts.train_nn', problem, architecture, *inputs, '$@',
                f'--num_features={2 if problem == "coal" else 1}']
        # Also save the mixture density network rather than just the compressor.
        if architecture == 'mdn_compressor':
            mdn = bb.File("mdn.pt")
            outputs.append(mdn)
            args.append(f'--mdn_output={mdn}')
        bb.Subprocess(outputs, inputs, args, env={'SEED': 1, 'LOGLEVEL': 'INFO'})

# Run the inference for all problems.
num_samples = 5000
for problem in ['benchmark', 'coal']:
    with bb.group_artifacts('workspace', problem):
        # Set up the list of viable methods for the problem.
        methods = ['naive', 'fearnhead', 'nunes', 'regressor', 'mdn_compressor', 'mdn']
        if problem == 'benchmark':
            methods.append('stan')

        # Iterate over the methods and construct the transformations for running the inference.
        inputs = [bb.File("data/train.pkl"), bb.File("data/test.pkl")]
        for method in methods:
            if method in {'mdn_compressor', 'regressor', 'mdn'}:
                model_path = bb.File(f'{method}.pt')
                inputs.append(model_path)
                algorithm = 'mdn' if method == 'mdn' else 'neural_compressor'
                extra_arg = '--cls_options={{"path": "%s"}}' % model_path
            else:
                algorithm = method
                if method == 'stan':
                    extra_arg = \
                        '--sample_options={{"keep_fits": true, "seed": 0, "adapt_delta": 0.99}}'
                else:
                    extra_arg = None
            args = ['$!', '-m', 'summaries.scripts.run_inference', problem, algorithm, *inputs[:2],
                    num_samples, '$@']
            if extra_arg:
                args.append(extra_arg)
            bb.Subprocess(f"samples/{method}.pkl", inputs, args)
