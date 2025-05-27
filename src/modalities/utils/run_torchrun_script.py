import os
import signal
import subprocess
import time


def run_torchrun_with_cleanup(torch_run_args: list[str], script_args: list[str]):
    """Starts a script with torchrun and cleans up the process group on exit.
    While for training, it is advised to run torchrun directly in the command line,
    this function is useful for profiling a set of configs with torchrun, as it
    allows to run each config of a grid search in a separate torchrun environment
    with a subsequent cleanup of the process group. This is important as the environment
    is in an undefined state in case of errors such as OOMs.

    Note that the process group is killed regardless of the exit code of the script to
    enforce that all processes are stopped, no zombies are left behind and all GPU memory
    gets released. A less aggressive cleanup did not release the GPU memory in some cases.

    With CTRL+C the process group can be killed directly by the user.

    Example torchrun single node command args on 4 ranks:
       ["--nproc_per_node", "4",
        "--nnodes", "1",
        "--node_rank", "0",
        "--rdzv_id", "0",
        "--rdzv_backend", "c10d",
        "--rdzv_endpoint", "localhost:0"]

    Args:
        torch_run_args (list[str]): The arguments to pass to torchrun.
        script_args (list[str]): The script path and its arguments.
    """

    torch_run = ["torchrun", *torch_run_args]

    print("[Launcher] Starting torchrun...")
    print(f"[Launcher] Command: {' '.join(torch_run)} {' '.join(script_args)}")
    proc = subprocess.Popen([*torch_run, *script_args], preexec_fn=os.setsid)  # start a new process group

    try:
        proc.wait()
        print("[Launcher] torchrun exited. Forcing cleanup of process group...")

        # Always kill process group regardless of exit code
        try:
            # more graceful, allowing process to cleanup cleanup
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            time.sleep(10)
            # immediate, forceful killing the process without cleanup
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            print("[Launcher] Process group already exited.")
        except Exception as e:
            print(f"[Launcher] Failed to kill process group: {e}")
    except KeyboardInterrupt:
        print("[Launcher] Interrupted. Killing process group...")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            time.sleep(10)
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception as e:
            print(f"[Launcher] Error while handling KeyboardInterrupt: {e}")
        raise
