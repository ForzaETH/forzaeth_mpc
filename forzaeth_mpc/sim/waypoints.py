from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Tuple, Union
import numpy as np


@dataclass(frozen=True)
class WpCols:
    """Column indices for the canonical waypoint array shape (N, 7)."""
    X: int = 0
    Y: int = 1
    VX: int = 2
    KAPPA: int = 3
    S: int = 4
    D_LEFT: int = 5
    D_RIGHT: int = 6


WP = WpCols()


def _estimate_s_from_xy(xy: np.ndarray) -> np.ndarray:
    """Cumulative arc-length s from XY polyline (simple forward differences)."""
    if len(xy) == 0:
        return np.zeros((0,), dtype=float)
    diffs = np.diff(xy, axis=0, prepend=xy[0:1])
    ds = np.hypot(diffs[:, 0], diffs[:, 1])
    s = np.cumsum(ds)
    s -= s[0]
    return s


def _finite_diff_curvature(xy: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Crude planar curvature estimate from XY using central differences."""
    n = len(xy)
    if n < 3:
        return np.zeros((n,), dtype=float)
    dx = np.gradient(xy[:, 0])
    dy = np.gradient(xy[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    den = np.power(dx * dx + dy * dy, 1.5) + eps
    kappa = (dx * ddy - dy * ddx) / den
    return kappa


def extract_waypoints_data(
    waypoints: Union[np.ndarray, Sequence[Tuple[float, float]]],
    *,
    fill_missing_with_estimates: bool = True
) -> np.ndarray:
    """
    Convert an array-like of waypoints into a canonical array of shape (N, 7)
    with columns [x_m, y_m, vx_mps, kappa_radpm, s_m, d_left, d_right].

    Accepted inputs:
      - np.ndarray with shape (N, 2..7) -> coerced to (N, 7) with sensible defaults
      - Sequence[Tuple[x, y]] -> treated as XY only

    If some fields are missing and `fill_missing_with_estimates=True`, this will:
      - compute s from XY
      - estimate kappa from XY
      - set vx_mps=0 when absent
      - set d_left/d_right=+inf when absent (i.e., no bound)
    """
    arr = np.asarray(waypoints, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Waypoints must be a 2D array-like.")
    n, m = arr.shape
    if m < 2:
        raise ValueError("Need at least columns [x, y].")
    # Build to (N,7)
    out = np.zeros((n, 7), dtype=float)
    out[:, WP.X:WP.Y + 1] = arr[:, 0:2]
    # Optional columns if present
    if m >= 3:
        out[:, WP.VX] = arr[:, 2]
    if m >= 4:
        out[:, WP.KAPPA] = arr[:, 3]
    if m >= 5:
        out[:, WP.S] = arr[:, 4]
    if m >= 6:
        out[:, WP.D_LEFT] = arr[:, 5]
    if m >= 7:
        out[:, WP.D_RIGHT] = arr[:, 6]
    # Defaults for missing
    out[:, WP.D_LEFT] = np.where(np.isfinite(out[:, WP.D_LEFT]), out[:, WP.D_LEFT], np.inf)
    out[:, WP.D_RIGHT] = np.where(np.isfinite(out[:, WP.D_RIGHT]), out[:, WP.D_RIGHT], np.inf)
    arr = out

    # Fill missing fields if requested
    if fill_missing_with_estimates:
        # If s is all zeros or not strictly increasing, recompute from XY
        if np.allclose(arr[:, WP.S], 0.0) or not np.all(np.diff(arr[:, WP.S]) >= 0.0):
            arr[:, WP.S] = _estimate_s_from_xy(arr[:, [WP.X, WP.Y]])

        # If kappa is all zeros -> estimate (crude) from XY
        if np.allclose(arr[:, WP.KAPPA], 0.0):
            arr[:, WP.KAPPA] = _finite_diff_curvature(arr[:, [WP.X, WP.Y]])

        # vx default stays as-is (0) if absent

        # Boundaries default to +inf (already set)
    return arr


def nearest_waypoint_index_by_s(waypoints: np.ndarray, s_query: float) -> int:
    """Return index of the waypoint whose s is closest to s_query (modulo track)."""
    s = np.asarray(waypoints[:, WP.S], dtype=float)
    if s.size == 0:
        raise ValueError("Empty waypoints.")
    # Normalize to [0, track_len)
    track_len = float(s[-1]) if s[-1] > 0 else float(s.max(initial=0.0))
    if track_len <= 0:
        # fallback: use linear index distance
        return int(np.argmin(np.abs(s - s_query)))
    s_mod = np.mod(s, track_len)
    s_q = s_query % track_len
    idx = int(np.argmin(np.abs(s_mod - s_q)))
    return idx


def track_length_from_waypoints(waypoints: np.ndarray) -> float:
    """
    Best-effort track length from s column; if not reliable, falls back to
    last s or to XY arc-length.
    """
    s = np.asarray(waypoints[:, WP.S], dtype=float)
    if s.size == 0:
        return 0.0
    if np.all(np.diff(s) >= 0.0) and s[-1] > 0.0:
        return float(s[-1])
    # fallback to XY curve length
    xy = waypoints[:, [WP.X, WP.Y]]
    ds = np.hypot(*(np.diff(xy, axis=0).T))
    return float(np.sum(ds))


def target_speed_at_s(waypoints: np.ndarray, s_query: float) -> float:
    """Convenience: nearest-neighbor target speed lookup by arc-length."""
    idx = nearest_waypoint_index_by_s(waypoints, s_query)
    return float(waypoints[idx, WP.VX])
