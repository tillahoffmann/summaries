from __future__ import annotations
from dataclasses import dataclass
import functools as ft
import numpy as np
from typing import Callable, Dict, Generic, Type, TypeVar, Union


T = TypeVar("T")
V = TypeVar("V")


@dataclass
class Container(Generic[T]):
    simulated: T
    observed: T


ContainerOrValue = Union[Container[T], T]
ParamDict = Dict[str, np.ndarray]


def ravel_param_dict(params: ParamDict, batch_dims: int = 0) -> np.ndarray:
    """
    Ravel a dictionary of parameters to obtain a tensor.

    Args:
        params: Dictionary of parameters.
        batch_dims: Number of batch dimensions to preserve.

    Returns:
        Tensor of parameters with `batch_dims + 1` dimensions. The last dimension is ordered by
        dictionary key.
    """
    values = []
    for key, value in sorted(params.items()):
        if value.ndim < batch_dims:
            raise ValueError(f"requested {batch_dims} batch dimensions but parameter {key} has "
                             f"shape {value.shape}")
        shape = (*value.shape[:batch_dims], -1)
        values.append(np.reshape(value, shape))
    return np.concatenate(values, axis=-1)


def maybe_apply_to_container(func: Callable[..., V]) -> Callable[..., ContainerOrValue[V]]:
    """
    Apply a function to each element of a container or a value as-is.
    """
    @ft.wraps(func)
    def _wrapped(*args, **kwargs) -> ContainerOrValue[V]:
        # Split up the arguments by element.
        any_container = False
        args_by_element = {}
        kwargs_by_element = {}

        for arg in args:
            if isinstance(arg, Container):
                simulated = arg.simulated
                observed = arg.observed
                any_container = True
            else:
                simulated = observed = arg
            args_by_element.setdefault("simulated", []).append(simulated)
            args_by_element.setdefault("observed", []).append(observed)

        for key, value in kwargs.items():
            if isinstance(value, Container):
                simulated = value.simulated
                observed = value.observed
                any_container = True
            else:
                simulated = observed = value
            kwargs_by_element.setdefault("simulated", {})[key] = simulated
            kwargs_by_element.setdefault("observed", {})[key] = observed

        # Apply to the values directly if there are no containers.
        if not any_container:
            return func(*args, **kwargs)

        # Apply to observed and simulated separately and construct a container.
        return Container(**{
            key: func(*args_by_element.get(key, []), **kwargs_by_element.get(key, {})) for key in
            ["simulated", "observed"]
        })
    return _wrapped


def maybe_apply_to_simulated(func: Callable[..., V]) -> Callable[..., V]:
    """
    Apply a function to the simulated element of a container or a value as-is. This function is
    "shallow" and only considers top-level arguments and not their members.
    """
    @ft.wraps(func)
    def _wrapped(*args, **kwargs) -> V:
        args = [arg.simulated if isinstance(arg, Container) else arg for arg in args]
        kwargs = {key: value.simulated if isinstance(value, Container) else value for key, value in
                  kwargs.items()}
        return func(*args, **kwargs)
    return _wrapped


def maybe_fit_simulated_transform_container(cls: Type[T]) -> Type[T]:
    """
    Fit a transformer to simulated data and apply it to both.
    """
    class _Transformer(cls):
        pass
    _Transformer.fit = maybe_apply_to_simulated(_Transformer.fit)
    _Transformer.transform = maybe_apply_to_container(_Transformer.transform)

    return _Transformer
