from unittest import mock
from summaries.scripts import util


def test_setup_with_seed():
    with mock.patch('os.environ.get') as env_get, mock.patch('numpy.random.seed') as np_seed, \
            mock.patch('torch.manual_seed') as th_seed, mock.patch('logging.basicConfig') as config:
        env_get.side_effect = ('debug', 23)
        util.setup()
        config.assert_called_once_with(level='DEBUG')
        np_seed.assert_called_once_with(23)
        th_seed.assert_called_once_with(23)
