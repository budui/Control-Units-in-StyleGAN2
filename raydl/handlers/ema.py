import ignite.distributed as idist
import torch
from ignite.handlers import Engine, Events


class ModelExponentialMovingAverage:
    def __init__(self, model, model_ema, num_items_of_half_life, batch_size_per_gpu, rampup=None,
                 num_items_transform=None):
        assert isinstance(model, torch.nn.Module)
        assert isinstance(model_ema, torch.nn.Module)
        assert isinstance(num_items_of_half_life, int) and num_items_of_half_life > 0
        assert num_items_transform is None or callable(num_items_transform)

        self.model = model
        self.model_ema = model_ema
        self.num_items_of_half_life = num_items_of_half_life
        self.rampup = rampup
        self.batch_size_per_gpu = batch_size_per_gpu
        self.num_items_transform = num_items_transform

    @staticmethod
    @torch.no_grad()
    def accumulate(model_ema, model, decay=0.999):
        for p_ema, p in zip(model_ema.parameters(), model.parameters()):
            p_ema.copy_(p.lerp(p_ema, decay))
        for b_ema, b in zip(model_ema.buffers(), model.buffers()):
            b_ema.copy_(b)

    def update(self, engine):
        if self.num_items_transform is None:
            num_items = engine.state.num_items
        else:
            num_items = self.num_items_transform(engine)
        world_size = engine.state.world_size if hasattr(engine.state, "world_size") else idist.get_world_size()
        ema_num_items = self.num_items_of_half_life
        if self.rampup is not None:
            ema_num_items = min(ema_num_items, num_items * self.rampup)
        # The half-life is the time lag at which the exponential weights decay by one half.
        # (ema_beta)^(num_items)=0.5
        ema_beta = 0.5 ** (self.batch_size_per_gpu * world_size / max(ema_num_items, 1e-8))
        self.accumulate(self.model_ema, self.model, decay=ema_beta)

    def init(self):
        # copy params from g to g_ema.
        # need to call this after load_state_dict
        # make sure g's params would not be updated after this and before the first iteration.
        self.accumulate(self.model_ema, self.model, 0)

    def attach(self, engine: Engine, event=Events.ITERATION_COMPLETED):
        engine.add_event_handler(Events.STARTED, self.init)
        engine.add_event_handler(event, self.update)
