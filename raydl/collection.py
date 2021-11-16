import pathlib
import re
from typing import Iterable, Any, Union, Tuple, Sequence, Optional

import torch

__all__ = [
    "tuple_of_type",
    "tuple_of_indices",
    "parse_indices_str",
    "AttributeDict",
    "describe_dict",
    "paired_indexes"
]


def tuple_of_type(data: Union[Iterable, Any], target_types: Union[type, Sequence[type]] = (int, float),
                  skip_unwanted_item: bool = True) -> Tuple:
    """
    return a tuple of target type from data
    :param data:
    :param target_types:
    :param skip_unwanted_item:
    :return:
    """
    if isinstance(data, target_types):
        return (data,)
    if isinstance(data, Iterable):
        values = []
        for i, d in enumerate(data):
            if not isinstance(d, target_types):
                if skip_unwanted_item:
                    continue
                else:
                    raise ValueError(f"expect types: {target_types}, but the type of the {i}-th items is {type(d)}")
            values.append(d)
        return tuple(values)
    raise ValueError(f"expect data have type {target_types} or Iterable of {target_types} but got {type(data)}")


def parse_indices_str(indices_str: str) -> Tuple:
    """
    parse string that contains indices
    :param indices_str: a string that only contains indices, i.e. only contains digit[0-9], `-`, and `,`
    :return: a tuple of indices, keep in order of indices_str
    """
    assert isinstance(indices_str, str), f"indices_str must be string, but got {type(indices_str)}"
    assert re.match(r"^(((\d+-\d+)|\d+),?)+$", indices_str) is not None, "invalid list string"

    if "," not in indices_str and "-" not in indices_str:
        return (int(indices_str),)
    result = []
    for sub in indices_str.split(","):
        if "-" in sub:
            s_left, s_right = tuple(map(int, sub.split("-")))
            delta = -1 if s_right < s_left else 1
            result += list(range(s_left, s_right + delta, delta))
        else:
            result.append(int(sub))
    return tuple(result)


def distinct_sequence(seq: Sequence):
    # if with python > 3.7, below must be faster.
    # although faster make a little a little sense.
    # https://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-whilst-preserving-order
    # return list(dict.fromkeys(items))
    seen = set()
    seen_add = seen.add
    return (x for x in seq if not (x in seen or seen_add(x)))


def tuple_of_indices(data: Union[int, Sequence[int]], need_distinct: bool = False):
    indices = parse_indices_str(data) if isinstance(data, str) else tuple_of_type(data, int)
    if not need_distinct:
        return indices
    return distinct_sequence(indices)


def paired_indexes(arg1, arg2):
    index1 = tuple_of_indices(arg1)
    index2 = tuple_of_indices(arg2)

    if len(index1) > 1:
        assert len(index2) == len(index1)

    if len(index1) == 1:
        index1 = index1 * len(index2)
    return index1, index2


class AttributeDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


def describe_dict(state_dict: dict, separator: str = "\n\t", head: Optional[str] = "dict content:") -> str:
    """
    return dict description str.
    """
    if not isinstance(state_dict, dict):
        raise ValueError(f"describe_dict only accept dict type, but got {type(state_dict)}")
    strings = [head] if head is not None else []
    for k, v in state_dict.items():
        if isinstance(v, (int, float, str, pathlib.Path)) or v is None:
            value_str = str(v)
        elif torch.is_tensor(v):
            if len(v.size()) == 0 or torch.numel(v) <= 16 and v.size(0) == 1:
                value_str = str(v)
            else:
                value_str = f"{type(v)}(dtype={v.dtype}, device={v.device}, shape={v.shape})"
        elif hasattr(v, "__len__"):
            value_str = f"{type(v)}(length={len(v)})"
        else:
            value_str = f"{type(v)}"
        strings.append(f"{k}: {value_str}")
    return separator.join(strings)
