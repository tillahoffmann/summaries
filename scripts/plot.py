import argparse
import importlib
import json
import matplotlib.style
import numpy as np


def __main__(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('plot_function', help='package.module:function of the plotting function')
    parser.add_argument('output', help='output path for the figure')
    parser.add_argument('--kwargs', type=json.loads, default={},
                        help='JSON dictionary of arguments to pass to the plotting function')
    parser.add_argument('--style', help='style file to use')
    parser.add_argument('--seed', type=int, help='random number generator seed')
    args = parser.parse_args(args)

    # Setup for plotting.
    if args.style:
        matplotlib.style.use(args.style)
    if args.seed is not None:
        np.random.seed(args.seed)

    # Load the plotting function.
    module, function = args.plot_function.split(':')
    module = importlib.import_module(module)
    function = getattr(module, function)

    # Plot and save.
    fig = function(**args.kwargs)
    fig.savefig(args.output)


if __name__ == '__main__':
    __main__()
