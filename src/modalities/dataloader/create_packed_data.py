import logging
import shutil
from pathlib import Path

from pydantic import FilePath

from modalities.config.component_factory import ComponentFactory
from modalities.config.config import load_app_config_dict
from modalities.config.instantiation_models import PackedDatasetComponentsInstantiationModel
from modalities.dataloader.packed_data_generator import PackedDataGenerator
from modalities.registry.components import COMPONENTS
from modalities.registry.registry import Registry

logger = logging.getLogger(__name__)


def pack_encoded_data(config_file_path: FilePath):
    """
    Utility to encode an indexed, large jsonl-file.

    (see also `create_index` for more information)
    Returns .pbin-file, which can be inserted into a training process directly
    and does not require its original jsonl-file or the respective index file anymore.
    """
    # TODO: if we want to use alternative entrypoints together with the ResolverRegistry,
    #  we can currently not rely on the existing class resolver.
    #  This is based on its connection to the overall `AppConfig`.
    #  One would requires an object of it to instantiate the ResolverRegistry.
    #  This could get resolved by implementing on own ResolverRegistry for each entrypoint or adapting the existing
    #  ResolverRegistry to work dynamically with any type-hinted config object from config.py.
    config = load_app_config_dict(config_file_path)

    # copy the config file to the src_path parent and append the original hash
    src_path = Path(config["settings"]["src_path"])
    src_path_has_hash_suffix = len(src_path.suffixes) > 1 and len(src_path.suffixes[0]) == 7
    if src_path_has_hash_suffix:
        hash_suffix = src_path.suffixes[0]
        config_file_name_with_hash = config_file_path.stem + hash_suffix + "".join(config_file_path.suffixes)
        shutil.copyfile(config_file_path, src_path.parent / config_file_name_with_hash)

    registry = Registry(COMPONENTS)
    component_factory = ComponentFactory(registry=registry)
    components: PackedDatasetComponentsInstantiationModel = component_factory.build_components(
        config_dict=config, components_model_type=PackedDatasetComponentsInstantiationModel
    )

    generator = PackedDataGenerator(
        components.settings.src_path,
        index_path=components.settings.index_path,
        tokenizer=components.tokenizer,
        eod_token=components.settings.eod_token,
        jq_pattern=components.settings.jq_pattern,
        number_of_processes=components.settings.num_cpus,
        processing_batch_size=components.settings.processing_batch_size,
        raw_samples_queue_size=components.settings.raw_samples_queue_size,
        processed_samples_queue_size=components.settings.processed_samples_queue_size,
    )
    generator.run(components.settings.dst_path)
