import argparse
import os
import shutil
import subprocess
from datetime import datetime
from os.path import isdir, isfile, join
from pathlib import Path

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
        raise Exception(f"Subproces run failed with command {command}.")


def main(cpu: bool = False, single_gpu: bool = False, multi_gpu: bool = False, devices: str = "0,1"):
    """
    Run tests on cpu, single gpu and multiple gpus

    Examples:
    python tests/tests.py: run CPU tests + single-GPU tests on GPU 0 + multi-GPU tests on GPUs 0/1
    python tests/tests.py --devices 4,5: same as above but using devices 4/5
    python tests/tests.py --cpu: run CPU tests
    python tests/tests.py --single-gpu --devices 3: run CPU tests + single-GPU tests on GPU 3
    python tests/tests.py --multi-gpu --devices 3,4: run multi-GPU tests on GPUs 3/4
    """
    # parse input
    if not cpu and not single_gpu and not multi_gpu:  # run all tests
        cpu, single_gpu, multi_gpu = True, True, True
    if single_gpu:  # cpu & single_gpu are both executed together via pytest
        cpu = True
    try:
        devices = [int(device) for device in devices.split(",")]  # e.g. '0,1' -> [0, 1]
    except ValueError:
        exit(f"ERROR! devices needs to be a string of comma-separated ints, e.g. '0,1'. Specified devices = {devices}")
    devices = devices if len(devices) <= 2 else devices[:2]  # use max 2 devices

    # check input
    if single_gpu and len(devices) < 1:
        exit(f"ERROR! Need at least 1 device to run single_gpu tests. Specified devices = {devices}")
    if multi_gpu and len(devices) < 2:
        exit(f"ERROR! Need 2 devices to run multi_gpu tests. Specified devices = {devices}")

    # start
    print(f"> TESTS ON          CPU: {cpu}")
    print(f"> TESTS ON   SINGLE GPU: {single_gpu} " + f"(device={devices[0] if single_gpu else None})")
    print(f"> TESTS ON MULTIPLE GPU: {multi_gpu} " + f"(devices={devices if multi_gpu else None})")

    # run cpu / single-gpu tests
    if cpu or single_gpu:
        print("\n=== RUN CPU & SINGLE-GPU TESTS ===" if single_gpu else "\n=== RUN CPU TESTS ===")
        command_unit_tests = (
            f"cd {_ROOT_DIR} && CUDA_VISIBLE_DEVICES={devices[0] if single_gpu else None} python -m pytest"
        )
        subprocess_run(command_unit_tests)

    # run multi-gpu tests
    if multi_gpu:
        # distributed tests
        print("\n=== RUN MULTI-GPU TESTS ===")
        run_distributed_tests_directory = _ROOT_DIR / "tests"
        run_distributed_tests_script = _ROOT_DIR / "tests" / "run_distributed_tests.sh"
        assert isfile(run_distributed_tests_script), f"ERROR! {run_distributed_tests_script} does not exist."
        command_end_to_end_tests = (
            f"cd {run_distributed_tests_directory}; bash run_distributed_tests.sh {devices[0]} {devices[1]} --no-cov"
        )
        subprocess_run(command_end_to_end_tests)

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
        command_getting_started_example += f"bash scripts/run_getting_started_example.sh {devices[0]} {devices[1]}"
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
            f"cd {run_warmstart_example_directory}; sh pre_train_and_warmstart.sh {devices[0]} {devices[1]}"
        )
        subprocess_run(command_warmstart_example)

    print("\n=== DONE ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument("--cpu", action=argparse.BooleanOptionalAction, help="Do cpu testing.")
    parser.add_argument("--single-gpu", action=argparse.BooleanOptionalAction, help="Do single GPU testing.")
    parser.add_argument("--multi-gpu", action=argparse.BooleanOptionalAction, help="Do multi GPU testing.")
    parser.add_argument("--devices", type=str, default="0,1")
    args = vars(parser.parse_args())
    args = {k: v if v is not None else False for k, v in args.items()}  # None -> False
    main(**args)
