"""ROS-free loader for stack_master's global_waypoints.json track files."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Tuple

import numpy as np

from .waypoints import WP, track_length_from_waypoints

_REPO_ROOT = Path(__file__).resolve().parents[2]

_REFLINE_KEY = {
    "iqp": "global_traj_wpnts_iqp",
    "mintime": "global_traj_wpnts_mintime",
}

# Column order matches WP (x_m, y_m, vx_mps, kappa_radpm, s_m, d_left, d_right).
_FIELDS = ("x_m", "y_m", "vx_mps", "kappa_radpm", "s_m", "d_left", "d_right")


def _resolve_path(track: str) -> Path:
    p = Path(track)
    if p.is_file():
        return p
    tracks_dir = Path(os.environ.get("FORZAETH_MPC_TRACKS", _REPO_ROOT / "tracks"))
    p = tracks_dir / f"{track}.json"
    if p.is_file():
        return p
    raise FileNotFoundError(f"Could not resolve track '{track}' (looked in {tracks_dir})")


def load_track(track: str, refline: str = "iqp") -> Tuple[np.ndarray, float]:
    """Load a track's waypoints straight from its global_waypoints.json.

    Returns the canonical (N,7) array (see waypoints.WP for columns) and the
    track length in meters.
    """
    path = _resolve_path(track)
    with open(path, "r") as f:
        d = json.load(f)
    wpnts = d[_REFLINE_KEY[refline]]["wpnts"]
    arr = np.array([[float(wp[k]) for k in _FIELDS] for wp in wpnts], dtype=float)
    return arr, track_length_from_waypoints(arr)


if __name__ == "__main__":
    waypoints, track_len = load_track("THETRACK")
    assert waypoints.shape[1] == 7
    assert np.all(np.diff(waypoints[:, WP.S]) >= 0.0), "s column must be non-decreasing"
    assert track_len > 0
    print(f"OK: THETRACK -> {waypoints.shape[0]} wpnts, track_len={track_len:.3f} m")
