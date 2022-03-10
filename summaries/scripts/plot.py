import argparse
import importlib
import json
import matplotlib.style
from .util import setup


def __main__(args=None):
    setup()
    parser = argparse.ArgumentParser()
    parser.add_argument('plot_function', help='package.module:function of the plotting function')
    parser.add_argument('output', help='output path for the figure')
    parser.add_argument('--kwargs', type=json.loads, default={},
                        help='JSON dictionary of arguments to pass to the plotting function')
    parser.add_argument('--style', help='style file to use')
    args = parser.parse_args(args)

    # Setup for plotting.
    if args.style:
        matplotlib.style.use(args.style)

    # Load the plotting function.
    module, function = args.plot_function.split(':')
    module = importlib.import_module(module)
    function = getattr(module, function)

    # Plot and save.
    fig = function(**args.kwargs)
    fig.savefig(args.output)


if __name__ == '__main__':
    __main__()
