from enum import Enum


class SelectiveActivationCheckpointingVariants(Enum):
    """Enum for the different activation checkpointing variants."""

    FULL_ACTIVATION_CHECKPOINTING = "full_activation_checkpointing"
    SELECTIVE_LAYER_ACTIVATION_CHECKPOINTING = "selective_layer_activation_checkpointing"
    SELECTIVE_OP_ACTIVATION_CHECKPOINTING = "selective_op_activation_checkpointing"
