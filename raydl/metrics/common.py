import numbers
from collections import OrderedDict
from typing import Callable, Union

import torch
from ignite.engine import Engine, Events
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce


class Collector(Metric):
    required_output_keys = None

    def __init__(
            self,
            output_transform: Callable = lambda x: x,
            reset_after_computing=True,
            device: Union[str, torch.device] = torch.device("cpu"),
    ):
        self._reset_after_computing = reset_after_computing

        super().__init__(output_transform=output_transform, device=device)

    @staticmethod
    def _clean_v(v):
        if not isinstance(v, (numbers.Number, torch.Tensor)):
            raise TypeError(f"Output should be a number or torch.Tensor, but given {type(v)}")
        if torch.is_tensor(v):
            v = v.detach()
            if v.ndim == 0 or (v.ndim == 1 and len(v) == 1):
                return v
            if v.ndim == 2 and v.size(-1) == 1:
                return v.sum()
            if v.ndim == 1 and len(v) > 1:
                return v.sum()
            raise TypeError(f"if output is tensor, must have size Nx1 or 1 or N or zero dim,"
                            f" but given size: {v.size()} dim: {v.dim()}")
        return v

    def _len_v(self, v):
        if not torch.is_tensor(v) or v.ndim == 0:
            return torch.ones([], device=self._device)
        return torch.as_tensor([len(v)], device=self._device)

    def _break_combined_tensor(self):
        if not (self.accumulator.dim() == self.num_examples.dim() == 1) \
                or not (len(self.accumulator) == len(self.num_examples) == len(self.keys)):
            raise RuntimeError(
                f"internal variables do not match in size: {self.accumulator.size()} {self.num_examples.size()}")

        return OrderedDict(zip(self.keys, self.accumulator)), OrderedDict(zip(self.keys, self.num_examples))

    @reinit__is_reduced
    def reset(self) -> None:
        self.accumulator = torch.tensor([], dtype=torch.float64, device=self._device)
        self.num_examples = torch.tensor([], dtype=torch.int32, device=self._device)
        self.keys = []

    @reinit__is_reduced
    def update(self, output: OrderedDict) -> None:
        cur_keys = OrderedDict([(k, None) for k in output.keys()])
        _cur_num_examples = []
        _cur_examples = []
        for k in self.keys:
            if k in cur_keys:
                cur_keys.pop(k)
            v = output.get(k, None)
            _cur_num_examples.append(torch.zeros([], device=self._device) if v is None else self._len_v(v))
            _cur_examples.append(torch.zeros([], device=self._device) if v is None else self._clean_v(v))
        if len(cur_keys) > 0:
            # new key-value pair
            for k in cur_keys:
                v = output[k]
                try:
                    _cur_examples.append(self._clean_v(v))
                    _cur_num_examples.append(self._len_v(v))
                except TypeError as e:
                    raise ValueError(f"Invalid value: {e}. key-value pair: {k}: {v}")
                self.keys.append(k)
            self.accumulator = torch.cat(
                [self.accumulator, torch.zeros([len(cur_keys)], dtype=torch.float64, device=self._device)])
            self.num_examples = torch.cat(
                [self.num_examples, torch.zeros([len(cur_keys)], dtype=torch.int32, device=self._device)])
        self.accumulator.add_(torch.as_tensor(_cur_examples, dtype=torch.float64, device=self._device))
        self.num_examples.add_(torch.as_tensor(_cur_num_examples, dtype=torch.int32, device=self._device))

    @torch.no_grad()
    def iteration_completed(self, engine: Engine) -> None:
        output = self._output_transform(engine.state.output)
        self.update(output)

    @sync_all_reduce("num_examples", "accumulator")
    def compute(self):
        _accumulator, _num_examples = self._break_combined_tensor()
        if self._reset_after_computing:
            self.reset()
        return {k: float(_accumulator[k] / _num_examples[k]) for k in _accumulator}

    def attach(
            self, engine: Engine, name: str, event_name: Events = Events.ITERATION_COMPLETED
    ) -> None:
        engine.add_event_handler(Events.STARTED, self.started)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)
        engine.add_event_handler(event_name, self.completed, name)

    @property
    def description(self):
        s = "Collector State:\n"
        s += f"\tMetric(num_examples):accumulator\n"
        for i, k in enumerate(self.keys):
            s += f"\t{k}({int(self.num_examples[i])}): {float(self.accumulator[i]):.4f}\n"
        return s
