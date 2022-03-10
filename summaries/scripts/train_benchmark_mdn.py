import argparse
import logging
import numpy as np
import os
import pickle
from summaries import benchmark
import torch as th
from .util import setup


def __main__(args: list[str] = None):
    logger = logging.getLogger('train_benchmark_mdn')
    setup()

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_features', help='number of summary statistics', type=int, default=1)
    parser.add_argument('--num_components', help='number of mixture components', type=int,
                        default=2)
    parser.add_argument('--max_epochs', help='maximum number of training epochs', type=int)
    parser.add_argument('--lr0', help='initial learning rate', type=float, default=0.01)
    parser.add_argument('--patience', help='number of epochs after which to terminate if the '
                        'validation loss does not decrease', type=int, default=10)
    parser.add_argument('--batch_size', help='minibatch size for training', type=int, default=1000)
    parser.add_argument('train', help='training data path')
    parser.add_argument('validation', help='validation data path')
    parser.add_argument('mdn_output', help='output path for the mixture density network')
    parser.add_argument('compressor_output', help='output for the compressor')
    args = parser.parse_args(args)

    # Load the data and create data loaders for training.
    paths = {'train': args.train, 'validation': args.validation}
    datasets = {}
    for key, path in paths.items():
        with open(path, 'rb') as fp:
            samples_ = pickle.load(fp)['samples']
        data = th.as_tensor(np.concatenate([samples_['x'], samples_['noise']], axis=-1))
        params = th.as_tensor(samples_['theta'][..., 0])
        dataset = th.utils.data.TensorDataset(data, params)
        datasets[key] = dataset
    data_loaders = {key: th.utils.data.DataLoader(dataset, args.batch_size, shuffle=True)
                    for key, dataset in datasets.items()}

    # Construct the MDN and learning rate schedule.
    mdn = benchmark.MDNBenchmarkAlgorithm(data.shape[-1], args.num_components, args.num_features)
    optimizer = th.optim.Adam(mdn.parameters(), args.lr0)
    scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.patience // 2)

    # Run the training.
    epoch = 0
    best_validation_loss = float('inf')
    num_bad_epochs = 0
    while num_bad_epochs < args.patience and (args.max_epochs is None or epoch < args.max_epochs):
        # Run one epoch using minibatches.
        train_loss = 0
        for x, theta in data_loaders['train']:
            dist: th.distributions.Distribution = mdn(x)
            loss: th.Tensor = - dist.log_prob(theta).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(data_loaders['train'])

        # Evaluate the validation loss and update the learning rate if required.
        validation_loss = - sum(mdn(x).log_prob(theta).mean().item() for x, theta
                                in data_loaders['validation']) / len(data_loaders['validation'])
        scheduler.step(validation_loss)

        # Update the best validation loss.
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            num_bad_epochs = 0
        else:
            num_bad_epochs += 1

        epoch += 1
        logger.info('epoch %3d: train loss = %.3f; validation loss = %.3f; best validation loss = '
                    '%.3f; # bad epochs = %d / %d', epoch, train_loss, validation_loss,
                    best_validation_loss, num_bad_epochs, args.patience)
    logger.info('training complete')

    # Save the results.
    os.makedirs(os.path.dirname(args.mdn_output), exist_ok=True)
    th.save(mdn, args.mdn_output)
    logger.info('saved MDN to %s', args.mdn_output)

    os.makedirs(os.path.dirname(args.compressor_output), exist_ok=True)
    th.save(mdn.compressor, args.compressor_output)
    logger.info('saved compressor to %s', args.mdn_output)


if __name__ == '__main__':
    __main__()
