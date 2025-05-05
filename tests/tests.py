import argparse
import os
import shutil
import subprocess
from datetime import datetime
from os.path import isdir, isfile, join
from pathlib import Path
from typing import Optional

import torch

_ROOT_DIR = Path(__file__).parents[1]


def check_existence_and_clear_getting_started_example_output(
    run_getting_started_example_directory: str, date_of_run: str
):
    # data
    output_directory_data = join(run_getting_started_example_directory, "data", "mem_map")
    output_files_data = [
        "redpajama_v2_samples_512_train.idx",
        "redpajama_v2_samples_512_test.idx",
        "redpajama_v2_samples_512_train.pbin",
        "redpajama_v2_samples_512_test.pbin",
    ]
    print()
    for output_file_data in output_files_data:
        output_file_path = join(output_directory_data, output_file_data)
        assert isfile(output_file_path), f"ERROR! {output_file_path} does not exist."
        try:
            os.remove(output_file_path)
            print(f"> removed {output_file_path}")
        except OSError as e:
            print(f"Error: {e.filename} - {e.strerror}.")

    # wandb directory
    output_directory_wandb = join(run_getting_started_example_directory, "data", "wandb_storage")
    assert isdir(output_directory_wandb), f"ERROR! {output_directory_wandb} does not exist"
    try:
        shutil.rmtree(output_directory_wandb)
        print(f"> removed {output_directory_wandb}")
    except OSError as e:
        print(f"Error: {e.filename} - {e.strerror}.")

    # checkpoint
    output_directory_checkpoints = join(run_getting_started_example_directory, "checkpoints")
    checkpoints = [elem for elem in os.listdir(output_directory_checkpoints) if elem.startswith("20")]
    checkpoint_to_delete = None
    for checkpoint in checkpoints:
        # e.g. "2024-08-14__09-54-53_abcde" -> "2024-08-14__09-54-53"
        date_of_checkpoint = "_".join(checkpoint.split("_")[:-1])
        if date_of_checkpoint > date_of_run:
            checkpoint_to_delete = join(output_directory_checkpoints, checkpoint)
            break
    assert checkpoint_to_delete is not None, f"ERROR! could not find a checkpoint with datetime > {date_of_run}"
    assert isdir(checkpoint_to_delete), f"ERROR! {checkpoint_to_delete} does not exist"
    try:
        shutil.rmtree(checkpoint_to_delete)
        print(f"> removed {checkpoint_to_delete}")
    except OSError as e:
        print(f"Error: {e.filename} - {e.strerror}.")

    # checkpoint converted
    checkpoints_converted = [
        join(output_directory_checkpoints, elem)
        for elem in os.listdir(output_directory_checkpoints)
        if elem.startswith("eid_")
    ]
    for checkpoint_converted in checkpoints_converted:
        assert isdir(checkpoint_converted), f"ERROR! {checkpoint_converted} does not exist"
        try:
            shutil.rmtree(checkpoint_converted)
            print(f"> removed {checkpoint_converted}")
        except OSError as e:
            print(f"Error: {e.filename} - {e.strerror}.")

    # config converted
    config_converted = join(run_getting_started_example_directory, "configs", "example_conversion_config.yaml")
    assert isfile(config_converted), f"ERROR! {config_converted} does not exist"
    try:
        os.remove(config_converted)
        print(f"> removed {config_converted}")
    except OSError as e:
        print(f"Error: {e.filename} - {e.strerror}.")


def get_checkpoint_from_getting_started_example(run_getting_started_example_directory: str) -> str:
    output_directory_checkpoints = join(run_getting_started_example_directory, "checkpoints")

    checkpoint_directories = [
        join(output_directory_checkpoints, elem)
        for elem in os.listdir(output_directory_checkpoints)
        if isdir(join(output_directory_checkpoints, elem))
    ]
    assert (
        len(checkpoint_directories) == 1
    ), f"ERROR! found {len(checkpoint_directories)} checkpoint directories for getting started example, expected 1."
    checkpoint_directory = checkpoint_directories[0]

    checkpoints = [
        join(checkpoint_directory, elem)
        for elem in os.listdir(checkpoint_directory)
        if isfile(join(checkpoint_directory, elem))
    ]
    checkpoints = [elem for elem in checkpoints if "model" in elem and elem.endswith(".bin")]
    assert (
        len(checkpoints) == 1
    ), f"ERROR! found {len(checkpoints)} checkpoints for getting started example, expected 1."
    checkpoint = checkpoints[0]

    return checkpoint


def replace_checkpoint_in_conversion_config(
    run_getting_started_example_directory: str, modalities_checkpoint: str
) -> str:
    # read example config
    example_config = join(run_getting_started_example_directory, "configs", "example_config.yaml")
    assert isfile(example_config), f"ERROR! could not find file at {example_config}"
    with open(example_config, "r") as f:
        lines = f.readlines()

    # read conversion config template
    conversion_config_template = join(
        run_getting_started_example_directory, "configs", "example_conversion_config_template.yaml"
    )
    assert isfile(conversion_config_template), f"ERROR! could not find file at {conversion_config_template}"
    with open(conversion_config_template, "r") as f:
        lines_additional = f.readlines()
    lines += lines_additional

    last_line_start = "    checkpoint_path:"
    assert lines[-1].startswith(
        last_line_start
    ), f"ERROR! expected file at {conversion_config_template} to contain 'checkpoint_path' in last line."
    lines[-1] = f"{last_line_start} {modalities_checkpoint}"

    # write conversion config
    conversion_config = join(run_getting_started_example_directory, "configs", "example_conversion_config.yaml")
    with open(conversion_config, "w") as f:
        for line in lines:
            f.write(line)
    return conversion_config


def subprocess_run(command: str) -> None:
    print(command)
    try:
        subprocess.run(command, shell=True, capture_output=False, check=True, text=True)
    except subprocess.CalledProcessError:
        raise Exception(f"Subprocess run failed with command {command}")


def main(
    include_main_tests: bool = False,
    include_torchrun_tests: bool = False,
    include_examples: bool = False,
    devices: Optional[str] = None,
    main_tests_name_filter: str = None,
) -> None:
    """
    Run tests and examples.
    There are two types of tests:
    - CPU and GPU (distributed and non-distributed) tests that don't require to be started via torchrun
        (included via include_main_tests)
    - distributed tests that require to be started via torchrun (included via include_torchrun_tests)
    If any of the main tests is not provided with enough GPUs, it will be skipped.

    There are several examples whose configs are being executed if include_examples is set to True.
    The examples are:
    - getting started example
    - checkpoint conversion example (based on getting started example)
    - warmstart example

    Examples:
    python tests/tests.py: Run no tests or examples
    python tests/tests.py --devices 4,5: same as above
    python tests/tests.py --include_main_tests: run the main tests
    python tests/tests.py --include_torchrun_tests: run the torchrun tests
    python tests/tests.py --include_examples: run the examples
    python tests/tests.py --include_main_tests --include_torchrun_tests --include_examples: run all tests and examples
    python tests/tests.py --include_main_tests --include_torchrun_tests --include_examples --devices 0,1,2,3:
        run all tests and examples on devices 0, 1, 2 and 3
    python tests/tests.py --include_main_tests --main_tests_name_filter test_initialization_fsdpx: run main tests that
        match specified filter `test_initialization_fsdpx`
    """
    if not any([include_main_tests, include_torchrun_tests, include_examples]):
        print(
            f"No tests selected to run ({include_main_tests=}, {include_torchrun_tests=}, "
            f"{include_examples=}). Exiting."
        )
        return

    if devices is None:
        device_ids = list(range(torch.cuda.device_count()))
    else:
        try:
            device_ids = devices.split(",")
        except ValueError:
            exit(
                "ERROR! devices needs to be a string of comma-separated ints, e.g. '0,1'. "
                f"Specified devices = {devices}"
            )

    # only run tests on max 4 devices
    device_ids = device_ids[:4]
    print("> Test setup: ")
    print(f"> {include_main_tests=}, {include_torchrun_tests=}, {include_examples=}")
    print(f"> {device_ids=}")

    if include_main_tests:
        # run cpu / gpu tests not requiring torchrun
        print(f"\n=== RUN MAIN TESTS on CPU and CUDA devices {device_ids} ===")
        command_unit_tests = (
            f"cd {_ROOT_DIR} && CUDA_VISIBLE_DEVICES="
            f"{','.join(map(str, device_ids)) if len(device_ids) >0 else ''} python -m pytest"
        )
        if main_tests_name_filter is not None:
            command_unit_tests += f" -k {main_tests_name_filter}"
        subprocess_run(command_unit_tests)

    if len(device_ids) < 2 and (include_torchrun_tests or include_examples):
        exit("ERROR! Need at least 2 devices to run torchrun tests and examples.")

    if include_torchrun_tests:
        # distributed tests
        print("\n=== RUN TORCHRUN TESTS ===")
        run_distributed_tests_directory = _ROOT_DIR / "tests"
        run_distributed_tests_script = _ROOT_DIR / "tests" / "run_distributed_tests.sh"
        assert isfile(run_distributed_tests_script), f"ERROR! {run_distributed_tests_script} does not exist."
        command_end_to_end_tests = (
            f"cd {run_distributed_tests_directory}; bash run_distributed_tests.sh "
            f"{' '.join(map(str, device_ids[:2]))} --no-cov"
        )
        subprocess_run(command_end_to_end_tests)

    if include_examples:
        # getting started example
        print("\n=== RUN GETTING STARTED EXAMPLE ===")
        run_getting_started_example_directory = _ROOT_DIR / "tutorials" / "getting_started"
        run_getting_started_example_script = (
            _ROOT_DIR / "tutorials" / "getting_started" / "scripts" / "run_getting_started_example.sh"
        )
        assert isfile(
            run_getting_started_example_script
        ), f"ERROR! {run_getting_started_example_script} does not exist."
        command_getting_started_example = f"cd {run_getting_started_example_directory}; "
        command_getting_started_example += (
            f"bash scripts/run_getting_started_example.sh {' '.join(map(str, device_ids[:2]))}"
        )
        date_of_run = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        subprocess_run(command_getting_started_example)

        # checkpoint conversion (based on getting started example)
        print("\n=== RUN CHECKPOINT CONVERSION (BASED ON GETTING STARTED EXAMPLE) ===")
        modalities_checkpoint = get_checkpoint_from_getting_started_example(run_getting_started_example_directory)
        conversion_config_path = replace_checkpoint_in_conversion_config(
            run_getting_started_example_directory, modalities_checkpoint
        )

        run_conversion_script = _ROOT_DIR / "tutorials" / "getting_started" / "scripts" / "run_checkpoint_conversion.sh"
        assert isfile(run_conversion_script), f"ERROR! {run_conversion_script} does not exist."
        command_conversion = f"cd {run_getting_started_example_directory}; "
        command_conversion += f"sh scripts/run_checkpoint_conversion.sh {conversion_config_path} "
        command_conversion += (
            f"{run_getting_started_example_directory}/checkpoints/{modalities_checkpoint.split('/')[-1]}"
        )
        subprocess_run(command_conversion)

        check_existence_and_clear_getting_started_example_output(run_getting_started_example_directory, date_of_run)

        # warmstart example
        print("\n=== RUN WARMSTART EXAMPLE ===")
        run_warmstart_example_directory = _ROOT_DIR / "tutorials/warmstart/scripts"
        run_warmstart_example_script = _ROOT_DIR / "tutorials/warmstart/scripts/pre_train_and_warmstart.sh"
        assert isfile(run_warmstart_example_script), f"ERROR! {run_warmstart_example_script} does not exist."
        command_warmstart_example = (
            f"cd {run_warmstart_example_directory}; sh pre_train_and_warmstart.sh {' '.join(map(str, device_ids[:2]))}"
        )
        subprocess_run(command_warmstart_example)

    print("\n=== DONE ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument(
        "-m",
        "--include_main_tests",
        action=argparse.BooleanOptionalAction,
        help="Run the main tests on CPU and GPU.",
        default=False,
    )
    parser.add_argument(
        "-t",
        "--include_torchrun_tests",
        action=argparse.BooleanOptionalAction,
        help="Run the tests that require to be launched in a torchrun environment.",
        default=False,
    )
    parser.add_argument(
        "-e",
        "--include_examples",
        action=argparse.BooleanOptionalAction,
        help="Run the examples.",
        default=False,
    )
    parser.add_argument("-d", "--devices", type=str, default=None)
    parser.add_argument("-f", "--main_tests_name_filter", type=str, default=None)
    args = vars(parser.parse_args())
    main(**args)
