from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful


class AppState(Stateful):
    """
    Note: this class has been copied from https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html

    This is a useful wrapper for checkpointing the Application State. Since this object is compliant
    with the Stateful protocol, DCP will automatically call state_dict/load_stat_dict as needed in the
    dcp.save/load APIs.

    Note: We take advantage of this wrapper to hande calling distributed state dict methods on the model
    and optimizer.
    """

    def __init__(self, model, optimizer=None):
        self._model = model
        self._optimizer = optimizer
        self._is_loaded = False

    @property
    def is_loaded(self):
        return self._is_loaded

    @property
    def model(self):
        return self._model

    @property
    def optimizer(self):
        return self._optimizer

    def state_dict(self):
        # this line automatically manages FSDP FQN's, as well as sets the default
        # state dict type to FSDP.SHARDED_STATE_DICT
        model_state_dict, optimizer_state_dict = get_state_dict(self._model, self._optimizer)
        return {"model": model_state_dict, "optim": optimizer_state_dict}

    def load_state_dict(self, state_dict):
        # sets our state dicts on the model and optimizer, now that we've loaded
        if self._is_loaded:
            raise RuntimeError(
                "Cannot call load_state_dict twice on the same AppState object. " "State dict has already been loaded."
            )
        set_state_dict(
            self._model, self._optimizer, model_state_dict=state_dict["model"], optim_state_dict=state_dict["optim"]
        )
        self._is_loaded = True
