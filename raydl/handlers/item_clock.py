import ignite.distributed as idist
from ignite.engine import Engine, Events, EventEnum


class TickEvent(EventEnum):
    TICK_STARTED = "tick_started"
    TICK_COMPLETED = "tick_completed"


class ItemClock:
    def __init__(
            self,
            num_items_per_tick=1024,
            batch_num_items_transform=lambda x: len(x),
    ):
        assert isinstance(num_items_per_tick, int) and num_items_per_tick > 0
        self.tick_size = num_items_per_tick
        self.batch_num_items_transform = batch_num_items_transform
        self.cur_tick = 1

    def _init(self, engine: Engine):
        if not hasattr(engine.state, "num_items"):
            engine.state.num_items = 0
        engine.state.tick = (engine.state.num_items + self.tick_size - 1) // self.tick_size

    def attach(self, engine: Engine):
        engine.state_dict_user_keys.append("num_items")
        # engine.state_dict_user_keys.append("tick")

        self._init(engine)
        engine.add_event_handler(Events.STARTED, self._init)

        @engine.on(Events.ITERATION_COMPLETED)
        def update_num_items(e: Engine):
            cur_items = self.batch_num_items_transform(e.state.batch)
            e.state.num_items += cur_items * idist.get_world_size()

        engine.register_events(
            *TickEvent,
            event_to_attr={
                TickEvent.TICK_STARTED: "tick",
                TickEvent.TICK_COMPLETED: "tick",
            }
        )

        @engine.on(Events.ITERATION_STARTED)
        def tick_start(e: Engine):
            s = e.state
            if (s.num_items + self.tick_size) // self.tick_size == s.tick + 1:
                s.tick += 1
                e.fire_event(TickEvent.TICK_STARTED)

        @engine.on(Events.ITERATION_COMPLETED)
        def tick_completed(e: Engine):
            s = e.state
            if s.num_items // self.tick_size == s.tick:
                e.fire_event(TickEvent.TICK_COMPLETED)
