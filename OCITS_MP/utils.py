import subprocess
import pandas as pd


def get_memory_usage():
    """Return current memory consumption of GPUS 1, 2 and 3. GPU 0 is not considered."""

    command = "nvidia-smi --query-gpu=memory.used --format=csv"

    out = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    stdout, _ = out.communicate()

    usage = str(stdout)[2:-1].split("\\n")[1:-1]
    usage = list(map(lambda x: int(x.replace(" MiB", "")), usage))
    usage = usage[1:]
    return usage
