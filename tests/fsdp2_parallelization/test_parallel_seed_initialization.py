import logging
import multiprocessing as py_mp
import os
import re
import traceback
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
from pydantic import BaseModel
from torch.distributed._tensor.placement_types import Replicate

from modalities.__main__ import Main
from modalities.batch import EvaluationResultBatch
from modalities.config.config import ProcessGroupBackendType
from modalities.config.pydantic_if_types import PydanticDeviceMeshIFType, PydanticFSDP2ModuleType
from modalities.logging_broker.messages import Message
from modalities.running_env.fsdp.device_mesh import ParallelismDegrees, get_parallel_rank
from tests.end2end_tests.custom_components import MultiProcessingCudaEnv
from tests.utility import monitor_child_processes

working_dir = Path(os.path.dirname(__file__))
tmp_folder = working_dir / "../tmp/fsdp2_warmstart_pp_tp"
working_dir = working_dir / "configs"


@pytest.mark.skipif(
    torch.cuda.device_count() < 8,
    reason="This e2e test requires 8 GPUs.",
)
class TestParallelSeedInitialization:
    WORLD_SIZE = 8
    RDVZ_PORT = 24574

    def test_parameters_follow_parallelism(self, tmp_path: Path):
        manager = py_mp.Manager()
        error_queue = manager.Queue()
        proc_ctx = mp.spawn(
            self._seed_distribution_impl_wrapper,
            args=(self.WORLD_SIZE, tmp_path, error_queue),
            nprocs=self.WORLD_SIZE,
            join=False,
        )
        monitor_child_processes(manager, error_queue, proc_ctx)

    def _seed_distribution_impl_wrapper(self, process_id: int, world_size: int, tmp_path: Path, error_queue: Any):
        with MultiProcessingCudaEnv(
            process_group_backend=ProcessGroupBackendType.nccl,
            global_rank=process_id,
            local_rank=process_id,
            world_size=world_size,
            rdvz_port=TestParallelSeedInitialization.RDVZ_PORT,
        ):
            try:
                self._seed_distribution_impl(world_size=world_size, tmp_path=tmp_path)
            except Exception as exc:
                tb = traceback.format_exc()
                logging.error(f"Process {process_id} (seed distribution test) encountered an error:\n{exc}")
                logging.error(tb)
                try:
                    error_queue.put((process_id, tb))
                except Exception:
                    logging.error("Failed to put exception info into error queue (seed distribution test).")
                os._exit(1)

    def _seed_distribution_impl(self, world_size: int, tmp_path: Path):
        # initialize components
        class ComponentsInstantiationModel(BaseModel):
            fsdp_model: PydanticFSDP2ModuleType
            device_mesh: PydanticDeviceMeshIFType

        config_file_path = self._get_tmp_sharding_config_path(dp_degree=2, tp_degree=2, pp_degree=2, tmp_path=tmp_path)
        main_obj = Main(config_file_path)
        components = main_obj.build_components(components_model_type=ComponentsInstantiationModel)
        model = components.fsdp_model
        device_mesh = components.device_mesh
        # for each pp stage get first transformer block's MLP weight parameter shards and full tensor
        block_key = next(iter(model.transformer.h.keys()))
        block = model.transformer.h[block_key]
        placements = [Replicate()] * len(block.mlp.W.weight.device_mesh.mesh.shape)
        full_weight = block.mlp.W.weight.redistribute(placements=placements).to_local().cpu()
        payload = {
            "tensor_full": full_weight,
            "tensor_shard": block.mlp.W.weight.to_local().cpu(),
            "tp_rank": get_parallel_rank(device_mesh=device_mesh, parallelism_method=ParallelismDegrees.TP),
            "pp_rank": get_parallel_rank(device_mesh=device_mesh, parallelism_method=ParallelismDegrees.PP),
            "dp_shard_rank": get_parallel_rank(device_mesh=device_mesh, parallelism_method=ParallelismDegrees.DP_SHARD),
            "block_key": block_key,
        }

        gather_list: list[dict[str, Any]] | None = [None] * world_size if dist.get_rank() == 0 else None
        dist.gather_object(payload, gather_list, dst=0)

        if dist.get_rank() == 0:
            assert gather_list is not None
            TestParallelSeedInitialization._assert_parameter_distribution(gather_list)
        dist.barrier()

    @staticmethod
    def _assert_parameter_distribution(records: list[dict[str, Any]]):
        combos: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for record in records:
            key = (record["pp_rank"], record["tp_rank"])
            combos.setdefault(key, []).append(record)

        expected_combo_count = 4
        assert (
            len(combos) == expected_combo_count
        ), f"Expected {expected_combo_count} PP/TP combinations, got {len(combos)}"

        combo_tensors: dict[tuple[int, int], torch.Tensor] = {}
        for (pp_rank, tp_rank), entries in combos.items():
            # check that full tensors are the same across data parallel processes
            reference = entries[0]["tensor_full"]
            seen_dp_ranks: set[int] = set()
            for entry in entries:
                dp_rank = entry["dp_shard_rank"]
                assert dp_rank not in seen_dp_ranks, f"Duplicate DP rank {dp_rank} for combo PP={pp_rank}, TP={tp_rank}"
                seen_dp_ranks.add(dp_rank)
                assert torch.equal(reference, entry["tensor_full"]), (
                    "Tensors within the same TP/PP combo must be identical across DP ranks; "
                    f"mismatch at DP rank {dp_rank} for (PP={pp_rank}, TP={tp_rank})"
                )
            # concatenate all shards for this pp/tp combo
            shards = sorted(entries, key=lambda e: e["dp_shard_rank"])
            combo_tensors[(pp_rank, tp_rank)] = torch.cat(
                [e["tensor_shard"] for e in shards],
                dim=0,
            )
        # check that tensor shards differ across different pp/tp combos
        combo_items = list(combo_tensors.items())
        for idx, ((pp_rank, tp_rank), base_tensor) in enumerate(combo_items):
            for other_key, other_tensor in combo_items[idx + 1 :]:
                tensors_equal = torch.equal(base_tensor, other_tensor)
                assert not tensors_equal, (
                    "Distinct TP/PP combinations should initialize with different weights; "
                    f"found match between (PP={pp_rank}, TP={tp_rank}) and (PP={other_key[0]}, TP={other_key[1]})"
                )

    def _get_tmp_sharding_config_path(self, dp_degree: int, tp_degree: int, pp_degree: int, tmp_path: Path) -> Path:
        temp_file_path = tmp_path / "pp_tp_sharding_config.yaml"
        working_dir = Path(os.path.dirname(__file__))
        config_file_path = (
            working_dir / "pipeline_parallelism/configs/config_lorem_ipsum_long_fsdp2_pp_tp_fwd_bwd_pass.yaml"
        )

        with open(config_file_path, "r") as file:
            config_string = file.read()
            config_dict = yaml.safe_load(config_string)
            config_dict["device_mesh"]["config"]["data_parallel_shard_degree"] = dp_degree
            config_dict["device_mesh"]["config"]["tensor_parallel_degree"] = tp_degree
            config_dict["device_mesh"]["config"]["pipeline_parallel_degree"] = pp_degree

        # save to temporary file
        with open(temp_file_path, "w") as file:
            yaml.dump(config_dict, file)

        return temp_file_path


def _get_loss_scores(messages: list[Message[EvaluationResultBatch]], loss_key: str) -> list[float]:
    return [message.payload.losses[loss_key].value.item() for message in messages]


def _extract_seen_steps_and_tokens(filename: str) -> tuple[int, int]:
    pattern = r"seen_steps_(\d+)-seen_tokens_(\d+)"
    match = re.search(pattern, filename)
    if match is None:
        raise ValueError(f"Filename '{filename}' does not match expected pattern '{pattern}'.")
    return int(match.group(1)), int(match.group(2))
