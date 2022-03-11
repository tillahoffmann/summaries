import numpy as np
import os
import pandas as pd
import pickle
from summaries.scripts import preprocess_coal
import tempfile


def test_preprocess_coal():
    with tempfile.TemporaryDirectory() as tmp:
        # Generate data and save it to disk.
        data_filename = os.path.join(tmp, 'data.csv')
        keys = ['theta', 'rho'] + [f'C{i}' for i in range(1, 8)]
        data = pd.DataFrame({key: np.random.normal(0, 1, 101) for key in keys})
        data.to_csv(data_filename, index=False)

        # Generate the splits.
        splits = {'foo.pkl': 20, 'bar.pkl': 81}
        preprocess_coal.__main__([data_filename, tmp] + [f'{f}={s}' for f, s in splits.items()])

        # Validate the splits.
        for filename, size in splits.items():
            with open(os.path.join(tmp, filename), 'rb') as fp:
                split = pickle.load(fp)

            assert split['samples']['theta'].shape == (size, 2)
            assert split['samples']['x'].shape == (size, 7)
