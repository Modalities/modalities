import os
import signal
import subprocess
import time


def run_torchrun_with_cleanup(torch_run_args: list[str], script_args: list[str]):
    """
    Example torchrun single node command args:
       ["--nproc_per_node", "4",
        "--nnodes", "1",
        "--rdzv_id", "0",
        "--rdzv_backend", "c10d",
        "--rdzv_endpoint", "localhost:0"]

    Args:
        torch_run_args (list[str]): _description_
        script_args (list[str]): _description_
    """

    torch_run = ["torchrun", *torch_run_args]

    print("[Launcher] Starting torchrun...")
    proc = subprocess.Popen([*torch_run, *script_args], preexec_fn=os.setsid)  # start a new process group

    try:
        proc.wait()
        print("[Launcher] torchrun exited. Forcing cleanup of process group...")

        # Always kill process group regardless of exit code
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            time.sleep(2)
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            print("[Launcher] Process group already exited.")
        except Exception as e:
            print(f"[Launcher] Failed to kill process group: {e}")
    except KeyboardInterrupt:
        print("[Launcher] Interrupted. Killing process group...")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            time.sleep(2)
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception as e:
            print(f"[Launcher] Error while handling KeyboardInterrupt: {e}")
        raise
