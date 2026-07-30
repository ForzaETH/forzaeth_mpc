import os
from pathlib import Path

_ENV_VAR = 'FORZAETH_MPC_CONFIG'

# repo_root/forzaeth_mpc/config/paths.py -> repo_root/config
_DEFAULT_CONFIG_ROOT = Path(__file__).resolve().parents[2] / 'config'


def config_root() -> Path:
    """Directory holding the per-car config dirs (NUC5/, DEFAULT/, ...).
    Override with the FORZAETH_MPC_CONFIG env var; on the car point it at
    $(rospack find stack_master)/config."""
    override = os.environ.get(_ENV_VAR)
    return Path(override) if override else _DEFAULT_CONFIG_ROOT
