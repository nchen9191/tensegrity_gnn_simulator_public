import shutil
from pathlib import Path

import torch

DEFAULT_DTYPE = torch.float32


def get_num_precision(precision: str) -> torch.dtype:
    if precision.lower() == 'float':
        return torch.float
    elif precision.lower() == 'float16':
        return torch.float16
    elif precision.lower() == 'float32':
        return torch.float32
    elif precision.lower() == 'float64':
        return torch.float64
    else:
        print("Precision unknown, defaulting to float64")
        return torch.float64


def save_curr_code(code_dir, output_dir):
    code_dir = Path(code_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(exist_ok=True)
    for p in code_dir.iterdir():
        if p.suffix in [".py", ".json", ".xml"]:
            shutil.copy(p, output_dir / p.name)
        elif p.is_dir() and not p.name.startswith("."):
            save_curr_code(p, output_dir / p.name)


def compute_num_steps(time_gap, dt, tol=1e-6):
    num_steps = int(time_gap // dt)
    gap = time_gap - dt * num_steps
    num_steps += 0 if gap < tol else 1
    # num_steps += int(math.ceil(time_gap - dt * num_steps))

    return num_steps
