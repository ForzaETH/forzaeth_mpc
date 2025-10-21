# loader.py
from __future__ import annotations
import numpy as np
from pathlib import Path
from .wp import WP
def load_waypoints_csv(path: str | Path) -> np.ndarray:
    arr = np.loadtxt(path, delimiter=",", dtype=float)
    if arr.ndim!=2 or arr.shape[1]<7:
        raise ValueError("Waypoints must have >=7 columns [x,y,v*,kappa,s,d_left,d_right].")
    return arr[:, :7].copy()
def track_length_from_wp(waypoints: np.ndarray) -> float:
    return float(waypoints[-1, WP.S] - waypoints[0, WP.S])
