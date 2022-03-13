import logging
import torch as th


th.set_default_dtype(th.float64)


def pytest_sessionfinish():
    logging.raiseExceptions = False
