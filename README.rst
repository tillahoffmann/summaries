summaries
=========

.. image:: https://github.com/tillahoffmann/summaries/actions/workflows/main.yml/badge.svg
  :target: https://github.com/tillahoffmann/summaries/actions/workflows/main.yml

This repository accompanies the preprint `"Minimizing the expected posterior entropy yields optimal summary statistics" <https://arxiv.org/abs/2206.02340>`_ and can be used to reproduce all figures and reported results.

You can reproduce the results in three steps:

1. Set up a new, clean python virtual environment (you can also skip this step to use your host python environment, but your mileage may vary).
2. Install all the requirements by running :code:`pip install -r requirements.txt`.
3. Generate all result files by running the following code from the command line.

.. code-block:: bash

  # Ignore the figures until we have generated all data files.
  doit ignore figures
  # Generate the data files (use -n to parallelize if desired).
  doit -n [number of cores]
  # Generate the figures and a summary file `workspace/figures/figures.html`.
  doit forget figures
  doit figures

You will find all figures in the folder :code:`workspace/figures` together with a HTML report :code:`workspace/figures/figures.html` that contains additional information. This process takes about 40 minutes on an M1 MacBook Pro when parallelizing across six cores.

The code has been tested with python 3.9 on macOS 12.4 (Monterey) running on Apple silicon and on Ubuntu 20.04. The :code:`summaries` package has complete test coverage, and you can run :code:`pytest` from the repository root to verify that your installation is working.
