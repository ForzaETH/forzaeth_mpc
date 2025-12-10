import hashlib

import numpy as np
import torch


def check_file_changes(
    files: list, hashfile_name: str, rewrite_hash=False, rewrite_hash_if_changed=True
):
    curr_hash = get_build_file_hashes(files)
    try:
        with open(hashfile_name, "r", encoding="utf-8") as hashfile:
            last_hash = hashfile.read()
            hash_changed = last_hash != curr_hash
    except FileNotFoundError:
        hash_changed = True

    if rewrite_hash or (hash_changed and rewrite_hash_if_changed):
        with open(hashfile_name, "w", encoding="utf-8") as hashfile:
            hashfile.write(curr_hash)

    return hash_changed


def get_build_file_hashes(files):
    solver_code = b""
    for file in files:
        with open(file, "rb") as gen_file:
            solver_code += gen_file.read()

    return hashlib.md5(solver_code).hexdigest()


def load_tensor_data(z_path: str, y_path: str):
    """
    Load offline data from CSV into float32 torch tensors.

    z_path: CSV with inputs, shape (N, nx+nu) or (N,)
    y_path: CSV with residuals, shape (N, nw) or (N,)
    """
    try:
        x_np = np.genfromtxt(z_path, delimiter=",")
        x_tensor = torch.from_numpy(x_np).type(torch.float32)
        if x_tensor.dim() == 1:
            x_tensor = x_tensor.unsqueeze(-1)
    except FileNotFoundError:
        print(f"[LBMPC] File {z_path} does not exist.")
        x_tensor = None

    try:
        y_np = np.genfromtxt(y_path, delimiter=",")
        y_tensor = torch.from_numpy(y_np).type(torch.float32)
        if y_tensor.dim() == 1:
            y_tensor = y_tensor.unsqueeze(-1)
    except FileNotFoundError:
        print(f"[LBMPC] File {y_path} does not exist.")
        y_tensor = None

    return x_tensor, y_tensor
