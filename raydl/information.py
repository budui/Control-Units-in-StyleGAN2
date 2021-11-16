import warnings
from typing import Sequence

import ignite
import ignite.distributed as idist
import torch
import torch.nn as nn

import raydl
from raydl import AttributeDict
from raydl.collection import describe_dict

__all__ = [
    "distributed_configure",
    "memory_usage",
    "module_summary",
    "library_version",
    "suppress_tracer_warnings",
    "class_repr",
]


def class_repr(module, attrs: Sequence[str], additional_items: dict = None):
    r_dict = {k: getattr(module, k) for k in attrs}
    if additional_items is not None:
        r_dict.update(additional_items)
    separator = "\n\t"
    r = raydl.describe_dict(r_dict, head=f"{module.__class__.__name__}(", separator=separator)
    return f"{r}\n)"


def module_summary(module: nn.Module, inputs: Sequence, max_nesting: int = 3, skip_redundant: bool = True) -> str:
    assert isinstance(module, torch.nn.Module)
    assert not isinstance(module, torch.jit.ScriptModule)
    assert isinstance(inputs, Sequence)

    # Register hooks.
    entries = []
    nesting = [0]  # use list to keep all module use the same nesting

    def pre_hook(_mod, _inputs):
        nesting[0] += 1

    def post_hook(mod, _inputs, outputs):
        nesting[0] -= 1
        if nesting[0] <= max_nesting:
            outputs = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
            outputs = [t for t in outputs if isinstance(t, torch.Tensor)]
            entries.append(AttributeDict(mod=mod, outputs=outputs))

    hooks = [mod.register_forward_pre_hook(pre_hook) for mod in module.modules()]
    hooks += [mod.register_forward_hook(post_hook) for mod in module.modules()]

    # Run module.
    outputs = module(*inputs)
    for hook in hooks:
        hook.remove()

    # Identify unique outputs, parameters, and buffers.
    tensors_seen = set()
    for e in entries:
        e.unique_params = [t for t in e.mod.parameters() if id(t) not in tensors_seen]
        e.unique_buffers = [t for t in e.mod.buffers() if id(t) not in tensors_seen]
        e.unique_outputs = [t for t in e.outputs if id(t) not in tensors_seen]
        tensors_seen |= {id(t) for t in e.unique_params + e.unique_buffers + e.unique_outputs}

    # Filter out redundant entries.
    if skip_redundant:
        entries = [e for e in entries if len(e.unique_params) or len(e.unique_buffers) or len(e.unique_outputs)]

    # Construct table.
    rows = [[type(module).__name__, 'Parameters', 'Buffers', 'Output shape', 'Datatype']]
    rows += [['---'] * len(rows[0])]
    param_total = 0
    buffer_total = 0
    submodule_names = {mod: name for name, mod in module.named_modules()}
    for e in entries:
        name = '<top-level>' if e.mod is module else submodule_names[e.mod]
        param_size = sum(t.numel() for t in e.unique_params)
        buffer_size = sum(t.numel() for t in e.unique_buffers)
        output_shapes = [str(list(e.outputs[0].shape)) for t in e.outputs]
        output_dtypes = [str(t.dtype).split('.')[-1] for t in e.outputs]
        rows += [[
            name + (':0' if len(e.outputs) >= 2 else ''),
            str(param_size) if param_size else '-',
            str(buffer_size) if buffer_size else '-',
            (output_shapes + ['-'])[0],
            (output_dtypes + ['-'])[0],
        ]]
        for idx in range(1, len(e.outputs)):
            rows += [[name + f':{idx}', '-', '-', output_shapes[idx], output_dtypes[idx]]]
        param_total += param_size
        buffer_total += buffer_size

    rows += [['---'] * len(rows[0])]
    rows += [['Total', str(param_total), str(buffer_total), '-', '-']]

    # Print table.
    widths = [max(len(cell) for cell in column) for column in zip(*rows)]
    summary_text = ""
    for row in rows:
        summary_text += '  '.join(cell + ' ' * (width - len(cell)) for cell, width in zip(row, widths))
        summary_text += '\n'
    return summary_text


def distributed_configure():
    return describe_dict({
        "distributed configuration": idist.model_name(),
        "backend": idist.backend(),
        "hostname": idist.hostname(),
        "world size": idist.get_world_size(),
        "rank": idist.get_rank(),
        "local rank": idist.get_local_rank(),
        "num nodes": idist.get_nnodes(),
        "num processes per_node": idist.get_nproc_per_node(),
        "node rank": idist.get_node_rank(),
    }, head="distributed_configure:")


def library_version():
    import torchvision
    versions = dict(
        PyTorch=torch.__version__,
        torchvision=torchvision.__version__,
        ignite=ignite.__version__,
    )
    if torch.cuda.is_available():
        # explicitly import cudnn as
        # torch.backends.cudnn can not be pickled with hvd spawning procs
        from torch.backends import cudnn

        versions["GPU"] = torch.cuda.get_device_name(idist.get_local_rank())
        versions["CUDA"] = torch.version.cuda
        versions["CUDNN"] = cudnn.version()
    return describe_dict(versions, head="version:")


def memory_usage(verbose=True):
    usage = "memory_usage:\n"
    memory = dict(
        max_reserverd_memory=torch.cuda.max_memory_reserved(),
        max_allocated_memory=torch.cuda.max_memory_allocated(),
    )
    torch.cuda.reset_peak_memory_stats()
    for k, v in memory.items():
        if v > 1024 ** 2:
            usage += f"{k}: {v / 1024 ** 2} MB\n"
        elif v > 1024:
            usage += f"{k}: {v / 1024} KB\n"
        else:
            usage += f"{k}: {v} B\n"
    if verbose:
        usage += torch.cuda.memory_summary()
    return usage


class suppress_tracer_warnings(warnings.catch_warnings):
    def __enter__(self):
        super().__enter__()
        warnings.simplefilter('ignore', category=torch.jit.TracerWarning)
        return self
