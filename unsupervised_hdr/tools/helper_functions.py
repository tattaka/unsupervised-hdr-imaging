from typing import Any, Dict, List, Union

import numpy as np
import torch

TORCH_MAJOR = int(torch.__version__.split(".")[0])
TORCH_MINOR = int(torch.__version__.split(".")[1])
if TORCH_MAJOR == 1 and TORCH_MINOR < 8:
    from torch._six import container_abcs, int_classes, string_classes
else:
    import collections.abc as container_abcs

    int_classes = int
    string_classes = str


def input2dict(input: Any, default_key: str) -> Dict[str, Any]:
    processed = {}
    if not isinstance(input, dict) and input is not None:
        processed[default_key] = input
        return processed
    else:
        return input


def input2list(input: Any) -> List[Any]:
    if input is not None:
        if not isinstance(input, list):
            processed = [input]
            return processed
        else:
            return input
    else:
        return []


def list2dict(input: List[Any]) -> Dict[str, Any]:
    if isinstance(input, list):
        output = {i: v for i, v in enumerate(input)}
        return output
    else:
        raise TypeError("input is not list!")


def concat_data(all_data):
    elem = all_data[0]
    # import sys
    # print(type(all_data[0][0]))
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in all_data])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        try:
            return torch.cat(all_data, out=out)
        except RuntimeError:
            return all_data
    elif (
        type(elem).__module__ == "numpy"
        and type(elem).__name__ != "str_"
        and type(elem).__name__ != "string_"
    ):
        if type(elem).__name__ == "ndarray":
            # array of string classes and object
            return np.concatenate(all_data)
        elif elem.shape == ():  # scalars
            return all_data
    elif isinstance(elem, float):
        return all_data
    elif isinstance(elem, int_classes):
        return all_data
    elif isinstance(elem, string_classes):
        return all_data
    elif isinstance(elem, container_abcs.Mapping):
        # [{1:torch.Tensor(), 2:torch.Tensor()}, {1:torch.Tensor(), 2:torch.Tensor()}...]みたいな時
        return {key: concat_data([d[key] for d in all_data]) for key in elem}
    elif isinstance(elem, container_abcs.Sequence):
        # [[[1, 2, 3, 4], [1, 2, 3, 4]], [[1, 2, 3, 4], [1, 2, 3, 4]]]
        # -> [[1, 2, 3, 4,] [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
        result = []
        [result.extend(d) for d in all_data]
        return result
