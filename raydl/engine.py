import time
from typing import Any, Callable

from ignite.engine import Engine, Events
from loguru import logger


class RayEngine(Engine):
    """
    add debug method for ignite.engine.Engine. if set debug as True, will also print event log.
    """

    def __init__(self, process_function: Callable, debug=False):
        super().__init__(process_function)
        self._firing_event_depth = 0
        self.debug = debug
        if self.debug:
            self.logger = logger.bind(event=True)

        self.add_event_handler(Events.STARTED, self._reset_firing_event_depth)

        self.state.attrs_to_log = []

    def _reset_firing_event_depth(self):
        self._firing_event_depth = 0

    @property
    def _current_status(self) -> str:
        if hasattr(self.state, "num_items"):
            return f"<num_items={self.state.num_items}>"

        last_event = self.last_event_name
        if last_event in self.state.event_to_attr:
            s = self.state
            name = self.state.event_to_attr[last_event]
            return f"<{name}={getattr(s, name)}>"
        else:
            return f"<iteration={self.state.iteration}>"

    def _fire_event(self, event_name: Any, *event_args: Any, **event_kwargs: Any) -> None:
        """
        reset logger info
        :param event_name:
        :param event_args:
        :param event_kwargs:
        :return:
        """
        if not self.debug:
            return super(RayEngine, self)._fire_event(event_name, *event_args, **event_kwargs)

        padding = " " * 6
        self.last_event_name = event_name
        self._firing_event_depth += 1
        start_time = None
        if len(self._event_handlers[event_name]):
            prefix = max(self._firing_event_depth - 2, 0) * f"│{padding}" + (
                1 if self._firing_event_depth >= 2 else 0) * f"{padding}├─ "
            self.logger.debug(f"{prefix}firing {event_name.name} at {self._current_status}")
            start_time = time.time()
        prefix = max(self._firing_event_depth - 1, 0) * f"{padding}│" + padding
        for func, args, kwargs in self._event_handlers[event_name]:
            kwargs.update(event_kwargs)
            first, others = ((args[0],), args[1:]) if (args and args[0] == self) else ((), args)
            self.logger.debug(f"{prefix}├─ {repr(func)}")
            func(*first, *(event_args + others), **kwargs)
        self._firing_event_depth -= 1
        if start_time is not None:
            overall_time = time.time() - start_time
            timer_str = f"{overall_time * 1000:.4f}ms" if overall_time * 1000 < 1 else f"{overall_time:.4f}s"
            self.logger.debug(f"{prefix}└── finish in {timer_str}")
