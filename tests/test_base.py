import numpy as np
import pytest
from summaries.base import Container, maybe_apply_to_container, maybe_apply_to_simulated, \
    ParamDict, ravel_param_dict
from typing import Any


def _add(a: float, b: float, c: float = 0) -> float:
    return a + b + c


@pytest.mark.parametrize("args, kwargs, expected", [
    [(3, 4), {}, 7],
    [(3, 4), {"c": Container(10, -10)}, Container(17, -3)],
    [(3, Container(3, -2)), {"c": 9}, Container(15, 10)],
])
def test_maybe_apply_to_container(args: tuple, kwargs: dict, expected: Any) -> None:
    assert maybe_apply_to_container(_add)(*args, **kwargs) == expected


@pytest.mark.parametrize("args, kwargs, expected", [
    [(3, 4), {}, 7],
    [(3, 4), {"c": Container(10, -10)}, 17],
    [(3, Container(3, -2)), {"c": 9}, 15],
])
def test_maybe_apply_to_simulated(args: tuple, kwargs: dict, expected: Any) -> None:
    assert maybe_apply_to_simulated(_add)(*args, **kwargs) == expected


a = np.random.normal(0, 1, (3, 4))
b = np.random.normal(0, 1, (3, 5, 7))


@pytest.mark.parametrize("x, batch_dims, expected", [
    [{"a": a[0], "b": b[0]}, 0, np.concatenate([a[0], b[0].ravel()])],
    [{"a": a, "b": b}, 1, np.hstack([a.reshape((3, -1)), b.reshape((3, -1))])],
])
def test_ravel_param_dict(x: ParamDict, batch_dims: int, expected: np.ndarray) -> None:
    np.testing.assert_array_equal(ravel_param_dict(x, batch_dims), expected)


def test_ravel_param_dict_invalid() -> None:
    with pytest.raises(ValueError, match="requested 1 batch dimensions but parameter b"):
        ravel_param_dict({"a": np.ones(3), "b": np.asarray(0)}, 1)
