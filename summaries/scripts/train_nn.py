import argparse
import logging
import pickle
import torch as th
from .. import benchmark, coal, nn, util


def evaluate_negative_log_likelihood(theta: th.Tensor, dist: th.distributions.Distribution) \
        -> th.Tensor:
    """
    Evaluate the negative log likelihood of a minibatch to minimize the expected posterior entropy
    directly.
    """
    loss = - dist.log_prob(theta)
    assert loss.shape == (theta.shape[0],)
    return loss.mean()


def __main__(args: list[str] = None):
    logger = logging.getLogger('train_nn')
    util.setup_script()

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_features', help='number of summary statistics', type=int, default=1)
    parser.add_argument('--num_components', help='number of mixture components', type=int,
                        default=2)
    parser.add_argument('--max_epochs', help='maximum number of training epochs', type=int)
    parser.add_argument('--lr0', help='initial learning rate', type=float, default=0.01)
    parser.add_argument('--patience', help='number of epochs after which to terminate if the '
                        'validation loss does not decrease', type=int, default=10)
    parser.add_argument('--batch_size', help='minibatch size for training', type=int, default=1000)
    parser.add_argument('--mdn_output', help='output path for the mixture density network')
    parser.add_argument('model', help='model whose parameters to infer',
                        choices=['benchmark', 'coal'])
    parser.add_argument('architecture', help='architecture of the neural network',
                        choices=['mdn_compressor', 'regressor'])
    parser.add_argument('train', help='training data path')
    parser.add_argument('validation', help='validation data path')
    parser.add_argument('output', help='output for the compressor model')
    args = parser.parse_args(args)

    # Load the data and create data loaders for training.
    paths = {'train': args.train, 'validation': args.validation}
    datasets = {}
    for key, path in paths.items():
        with open(path, 'rb') as fp:
            samples = pickle.load(fp)['samples']
        data = th.as_tensor(samples['x'])
        params = th.as_tensor(samples['theta'])
        dataset = th.utils.data.TensorDataset(data, params)
        datasets[key] = dataset
    data_loaders = {key: th.utils.data.DataLoader(dataset, args.batch_size, shuffle=True)
                    for key, dataset in datasets.items()}

    # Construct the compression module to extract features.
    if args.model == 'benchmark':
        compressor = nn.DenseCompressor([data.shape[-1], 16, 16, args.num_features], th.nn.Tanh())
    elif args.model == 'coal':
        compressor = nn.DenseStack([data.shape[-1], 16, 16, args.num_features], th.nn.Tanh())
    else:
        raise NotImplementedError(args.model)

    # Construct the neural network and loss function.
    if args.architecture == 'mdn_compressor':
        loss_function = evaluate_negative_log_likelihood
        if args.model == 'benchmark':
            expansion_nodes = [args.num_features, 16, args.num_components]
            module = benchmark.MixtureDensityNetwork(compressor, expansion_nodes, th.nn.Tanh())
        elif args.model == 'coal':
            expansion_nodes = [args.num_features, 16, args.num_components]
            module = coal.MixtureDensityNetwork(compressor, expansion_nodes, th.nn.Tanh())
        else:
            raise NotImplementedError(args.model)
    elif args.architecture == 'regressor':
        assert args.mdn_output is None, 'cannot save MDN module using regressor architecture'
        loss_function = th.nn.MSELoss()
        # We directly use the compressed statistics as predictors for the parameters.
        module = compressor
    else:
        raise NotImplementedError(args.architecture)

    optimizer = th.optim.Adam(module.parameters(), args.lr0)
    scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.patience // 2)

    # Run the training.
    epoch = 0
    best_validation_loss = float('inf')
    num_bad_epochs = 0
    while num_bad_epochs < args.patience and (args.max_epochs is None or epoch < args.max_epochs):
        # Run one epoch using minibatches.
        train_loss = 0
        for step, (x, theta) in enumerate(data_loaders['train']):
            y = module(x)
            loss: th.Tensor = loss_function(theta, y)
            assert not loss.isnan()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= step + 1

        # Evaluate the validation loss and update the learning rate if required.
        validation_loss = sum(loss_function(theta, module(x)).mean().item() for x, theta
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
    th.save(compressor, args.output)
    logger.info('saved compressor to %s', args.output)

    if args.mdn_output:
        th.save(module, args.mdn_output)
        logger.info('saved MDN to %s', args.mdn_output)


if __name__ == '__main__':
    __main__()
