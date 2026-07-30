import hashlib
import os

import numpy as np
import torch


def _ensure_parent_dir(file_path: str) -> None:
    parent_dir = os.path.dirname(file_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def check_file_changes(
    files: list, hashfile_name: str, rewrite_hash=False, rewrite_hash_if_changed=True
):
    curr_hash = get_build_file_hashes(files)
    _ensure_parent_dir(hashfile_name)
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


def _state_dict_to_cpu(state_dict: dict) -> dict:
    cpu_state_dict = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            cpu_state_dict[key] = value.detach().cpu()
        else:
            cpu_state_dict[key] = value
    return cpu_state_dict


def save_gp_model_bundle(model_path: str, gp_model, metadata: dict = None) -> None:
    train_x = None
    train_inputs = getattr(gp_model, "train_inputs", None)
    if train_inputs is not None and len(train_inputs) > 0:
        train_x = train_inputs[0]

    train_y = getattr(gp_model, "train_targets", None)

    bundle = {
        "gp_model_state_dict": _state_dict_to_cpu(gp_model.state_dict()),
        "likelihood_state_dict": _state_dict_to_cpu(gp_model.likelihood.state_dict()),
        "train_x": train_x.detach().cpu() if torch.is_tensor(train_x) else train_x,
        "train_y": train_y.detach().cpu() if torch.is_tensor(train_y) else train_y,
        "metadata": metadata or {},
    }

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    tmp_path = f"{model_path}.tmp"
    torch.save(bundle, tmp_path)
    os.replace(tmp_path, model_path)


def load_gp_model_bundle(model_path: str):
    if not os.path.exists(model_path):
        return None
    try:
        return torch.load(model_path, map_location="cpu")
    except Exception as exc:
        print(f"[LBMPC] Failed to load GP model bundle from {model_path}: {exc}")
        return None
