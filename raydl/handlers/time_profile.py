from collections import OrderedDict
from typing import Any, Dict, List, Mapping, Tuple, Union

import torch
import torch.nn.functional
from ignite.contrib.handlers import HandlersTimeProfiler as _HandlersTimeProfiler
from ignite.engine import Engine, EventEnum, Events
from ignite.handlers import Timer

from raydl.handlers import RunningStatistician


class HandlersTimeProfiler(_HandlersTimeProfiler):
    @staticmethod
    def results_string(results: List[List[Union[str, float]]]) -> str:
        # adopted implementation of torch.autograd.profiler.build_table
        handler_column_width = max([len(item[0]) for item in results]) + 4  # type: ignore[arg-type]
        event_column_width = max([len(item[1]) for item in results]) + 4  # type: ignore[arg-type]

        DEFAULT_COLUMN_WIDTH = 14

        headers = [
            "Handler",
            "Event Name",
            "Total(s)",
            "Min(s)/IDX",
            "Max(s)/IDX",
            "Mean(s)",
            "Std(s)",
        ]

        # Have to use a list because nonlocal is Py3 only...
        SPACING_SIZE = 2
        row_format_lst = [""]
        header_sep_lst = [""]
        line_length_lst = [-SPACING_SIZE]

        def add_column(padding: int, text_dir: str = ">") -> None:
            row_format_lst[0] += "{: " + text_dir + str(padding) + "}" + (" " * SPACING_SIZE)
            header_sep_lst[0] += "-" * padding + (" " * SPACING_SIZE)
            line_length_lst[0] += padding + SPACING_SIZE

        add_column(handler_column_width, text_dir="<")
        add_column(event_column_width, text_dir="<")
        for _ in headers[2:]:
            add_column(DEFAULT_COLUMN_WIDTH)

        row_format = row_format_lst[0]
        header_sep = header_sep_lst[0]

        result = []

        def append(s: str) -> None:
            result.append(s)
            result.append("\n")

        result.append("\n")
        append(header_sep)
        append(row_format.format(*headers))
        append(header_sep)

        for row in results[:-3]:
            # format min/idx and max/idx
            row[3] = "{}/{}".format(*row[3])  # type: ignore[misc]
            row[4] = "{}/{}".format(*row[4])  # type: ignore[misc]

            append(row_format.format(*row))

        append(header_sep)
        # print total handlers time row
        append(row_format.format(*results[-3]))
        append(header_sep)

        summary_format = "{} took total {}s [min/index: {}, max/index: {}, mean: {}s, std: {}s]"
        for row in results[-2:]:
            row[3] = "{}s/{}".format(*row[3])  # type: ignore[misc]
            row[4] = "{}s/{}".format(*row[4])  # type: ignore[misc]
            del row[1]
            append(summary_format.format(*row))

        return "".join(result)


class BasicTimeProfiler:
    """
    BasicTimeProfiler can be used to profile the handlers,
    events, data loading and data processing times.
    """

    events_to_ignore = [
        Events.EXCEPTION_RAISED,
        Events.TERMINATE,
        Events.TERMINATE_SINGLE_EPOCH,
        Events.DATALOADER_STOP_ITERATION,
    ]

    def __init__(self) -> None:
        self._dataflow_timer = Timer()
        self._processing_timer = Timer()
        self._event_handlers_timer = Timer()

        self.dataflow_times = RunningStatistician()
        self.processing_times = RunningStatistician()
        self.event_handlers_times = {}  # type: Dict[EventEnum, torch.Tensor]

        self._events = [
            Events.EPOCH_STARTED,
            Events.EPOCH_COMPLETED,
            Events.ITERATION_STARTED,
            Events.ITERATION_COMPLETED,
            Events.GET_BATCH_STARTED,
            Events.GET_BATCH_COMPLETED,
            Events.COMPLETED,
        ]
        self._fmethods = [
            self._as_first_epoch_started,
            self._as_first_epoch_completed,
            self._as_first_iter_started,
            self._as_first_iter_completed,
            self._as_first_get_batch_started,
            self._as_first_get_batch_completed,
            self._as_first_completed,
        ]
        self._lmethods = [
            self._as_last_epoch_started,
            self._as_last_epoch_completed,
            self._as_last_iter_started,
            self._as_last_iter_completed,
            self._as_last_get_batch_started,
            self._as_last_get_batch_completed,
            self._as_last_completed,
        ]

    def reset(self):
        self._reset()

    def _reset(self) -> None:
        self.dataflow_times = RunningStatistician()
        self.processing_times = RunningStatistician()
        self.event_handlers_times = {
            Events.STARTED: RunningStatistician(),
            Events.COMPLETED: RunningStatistician(),
            Events.EPOCH_STARTED: RunningStatistician(),
            Events.EPOCH_COMPLETED: RunningStatistician(),
            Events.ITERATION_STARTED: RunningStatistician(),
            Events.ITERATION_COMPLETED: RunningStatistician(),
            Events.GET_BATCH_COMPLETED: RunningStatistician(),
            Events.GET_BATCH_STARTED: RunningStatistician(),
        }

    def _as_first_started(self, engine: Engine) -> None:
        self._reset()

        self.event_handlers_names = {
            e: [
                h.__qualname__ if hasattr(h, "__qualname__") else h.__class__.__name__
                for (h, _, _) in engine._event_handlers[e]
                if "BasicTimeProfiler." not in repr(h)  # avoid adding internal handlers into output
            ]
            for e in Events
            if e not in self.events_to_ignore
        }

        # Setup all other handlers:
        engine._event_handlers[Events.STARTED].append((self._as_last_started, (engine,), {}))

        for e, m in zip(self._events, self._fmethods):
            engine._event_handlers[e].insert(0, (m, (engine,), {}))

        for e, m in zip(self._events, self._lmethods):
            engine._event_handlers[e].append((m, (engine,), {}))

        # Let's go
        self._event_handlers_timer.reset()

    def _as_last_started(self, engine: Engine) -> None:
        self.event_handlers_times[Events.STARTED].update(self._event_handlers_timer.value())

    def _as_first_epoch_started(self, engine: Engine) -> None:
        self._event_handlers_timer.reset()

    def _as_last_epoch_started(self, engine: Engine) -> None:
        self.event_handlers_times[Events.EPOCH_STARTED].update(self._event_handlers_timer.value())

    def _as_first_get_batch_started(self, engine: Engine) -> None:
        self._event_handlers_timer.reset()
        self._dataflow_timer.reset()

    def _as_last_get_batch_started(self, engine: Engine) -> None:
        t = self._event_handlers_timer.value()
        self.event_handlers_times[Events.GET_BATCH_STARTED].update(t)

    def _as_first_get_batch_completed(self, engine: Engine) -> None:
        self._event_handlers_timer.reset()

    def _as_last_get_batch_completed(self, engine: Engine) -> None:
        self.event_handlers_times[Events.GET_BATCH_COMPLETED].update(self._event_handlers_timer.value())

        self.dataflow_times.update(self._dataflow_timer.value())

        self._dataflow_timer.reset()

    def _as_first_iter_started(self, engine: Engine) -> None:
        self._event_handlers_timer.reset()

    def _as_last_iter_started(self, engine: Engine) -> None:
        self.event_handlers_times[Events.ITERATION_STARTED].update(self._event_handlers_timer.value())

        self._processing_timer.reset()

    def _as_first_iter_completed(self, engine: Engine) -> None:
        self.processing_times.update(self._processing_timer.value())

        self._event_handlers_timer.reset()

    def _as_last_iter_completed(self, engine: Engine) -> None:
        self.event_handlers_times[Events.ITERATION_COMPLETED].update(self._event_handlers_timer.value())

    def _as_first_epoch_completed(self, engine: Engine) -> None:
        self._event_handlers_timer.reset()

    def _as_last_epoch_completed(self, engine: Engine) -> None:
        self.event_handlers_times[Events.EPOCH_COMPLETED].update(self._event_handlers_timer.value())

    def _as_first_completed(self, engine: Engine) -> None:
        self._event_handlers_timer.reset()

    def _as_last_completed(self, engine: Engine) -> None:
        self.event_handlers_times[Events.COMPLETED].update(self._event_handlers_timer.value())

        # Remove added handlers:
        engine.remove_event_handler(self._as_last_started, Events.STARTED)

        for e, m in zip(self._events, self._fmethods):
            engine.remove_event_handler(m, e)

        for e, m in zip(self._events, self._lmethods):
            engine.remove_event_handler(m, e)

    def attach(self, engine: Engine) -> None:
        """Attach BasicTimeProfiler to the given engine.

        Args:
            engine: the instance of Engine to attach
        """
        if not isinstance(engine, Engine):
            raise TypeError(f"Argument engine should be ignite.engine.Engine, but given {type(engine)}")

        if not engine.has_event_handler(self._as_first_started):
            engine._event_handlers[Events.STARTED].insert(0, (self._as_first_started, (engine,), {}))

    @staticmethod
    def _compute_basic_stats(rs: RunningStatistician) -> Dict[
        str, Union[str, float, Tuple[Union[float], Union[float]]]]:
        out = [
            ("total", rs.sum().item() if len(rs) > 0 else "not yet triggered")
        ]  # type: List[Tuple[str, Union[str, float, Tuple[Union[float], Union[float]]]]]
        names = ["min/index", "max/index", "mean", "std"]
        if len(rs) > 1:
            statistics = rs.compute()
            for n in names:
                out.append((n, statistics[n]))
        return OrderedDict(out)

    def get_results(self) -> Dict[str, Dict[str, Any]]:
        """
        Method to fetch the aggregated profiler results after the engine is run

        .. code-block:: python

            results = profiler.get_results()

        """
        total_eh_time = sum(
            [(self.event_handlers_times[e]).sum() for e in Events if e not in self.events_to_ignore]
        )  # type: Union[int, torch.Tensor]
        event_handlers_stats = dict(
            [
                (str(e.name).replace(".", "_"), self._compute_basic_stats(self.event_handlers_times[e]))
                for e in Events
                if e not in self.events_to_ignore
            ]
            + [("total_time", total_eh_time)]  # type: ignore[list-item]
        )

        return OrderedDict(
            [
                ("processing_stats", self._compute_basic_stats(self.processing_times)),
                ("dataflow_stats", self._compute_basic_stats(self.dataflow_times)),
                ("event_handlers_stats", event_handlers_stats,),
                (
                    "event_handlers_names",
                    {str(e.name).replace(".", "_") + "_names": v for e, v in self.event_handlers_names.items()},
                ),
            ]
        )

    @staticmethod
    def results_string(results: Dict) -> str:
        """
        Method to print the aggregated results from the profiler

        Args:
            results: the aggregated results from the profiler

        .. code-block:: python

            profiler.results_string(results)

        Example output:

        .. code-block:: text

             ----------------------------------------------------
            | Time profiling stats (in seconds):                 |
             ----------------------------------------------------
            total  |  min/index  |  max/index  |  mean  |  std

            Processing function:
            157.46292 | 0.01452/1501 | 0.26905/0 | 0.07730 | 0.01258

            Dataflow:
            6.11384 | 0.00008/1935 | 0.28461/1551 | 0.00300 | 0.02693

            Event handlers:
            2.82721

            - Events.STARTED: []
            0.00000

            - Events.EPOCH_STARTED: []
            0.00006 | 0.00000/0 | 0.00000/17 | 0.00000 | 0.00000

            - Events.ITERATION_STARTED: ['PiecewiseLinear']
            0.03482 | 0.00001/188 | 0.00018/679 | 0.00002 | 0.00001

            - Events.ITERATION_COMPLETED: ['TerminateOnNan']
            0.20037 | 0.00006/866 | 0.00089/1943 | 0.00010 | 0.00003

            - Events.EPOCH_COMPLETED: ['empty_cuda_cache', 'training.<locals>.log_elapsed_time', ]
            2.57860 | 0.11529/0 | 0.14977/13 | 0.12893 | 0.00790

            - Events.COMPLETED: []
            not yet triggered

        """

        def to_str(v: Union[str, tuple]) -> str:
            if isinstance(v, str):
                return v
            elif isinstance(v, tuple):
                return f"{v[0]:.5f}/{v[1]}"
            return f"{v:.5f}"

        def odict_to_str(d: Mapping) -> str:
            out = " | ".join([to_str(v) for v in d.values()])
            return out

        others = {
            k: odict_to_str(v) if isinstance(v, OrderedDict) else v for k, v in results["event_handlers_stats"].items()
        }

        others.update(results["event_handlers_names"])

        output_message = """
 ----------------------------------------------------
| Time profiling stats (in seconds):                 |
 ----------------------------------------------------
total  |  min/index  |  max/index  |  mean  |  std

Processing function:
{processing_stats}

Dataflow:
{dataflow_stats}

Event handlers:
{total_time:.5f}

- Events.STARTED: {STARTED_names}
{STARTED}

- Events.EPOCH_STARTED: {EPOCH_STARTED_names}
{EPOCH_STARTED}

- Events.ITERATION_STARTED: {ITERATION_STARTED_names}
{ITERATION_STARTED}

- Events.ITERATION_COMPLETED: {ITERATION_COMPLETED_names}
{ITERATION_COMPLETED}

- Events.EPOCH_COMPLETED: {EPOCH_COMPLETED_names}
{EPOCH_COMPLETED}

- Events.COMPLETED: {COMPLETED_names}
{COMPLETED}
""".format(
            processing_stats=odict_to_str(results["processing_stats"]),
            dataflow_stats=odict_to_str(results["dataflow_stats"]),
            **others,
        )
        return output_message
