# plotting.py
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np

from ...mpc_core.utils.indices import IDX


def plot_three(
    X_mpc: np.ndarray,
    X_nom: np.ndarray,
    X_nomres: Optional[np.ndarray] = None,
    *,
    ref_xy: np.ndarray,          # (M,2) centerline [x,y]
    track_left: np.ndarray,      # (M,) left distances from centerline
    track_right: np.ndarray,     # (M,) right distances from centerline
    idx_s: Union[int, str] = 0,        # index (or IDX key) for s in state vector
    idx_n: Union[int, str] = 1,        # index (or IDX key) for n in state vector
    names=("vx", "vy", "wz"),    # time-series vars to plot (use IDX keys)
    titles: Optional[dict] = None,
):
    """
    Left:  Track (centerline + boundaries) and trajectories (nominal, actual, optional corrected),
           with states given in Frenet (s,n) converted to Cartesian (x,y).
    Right: 3 time-series plots for selected state names (via global IDX).
    """
    # ----------------- helpers -----------------
    def _get_idx(x):
        if isinstance(x, (int, np.integer)):
            return int(x)
        # assume it's a key into global IDX
        return int(IDX[x])

    def _compute_unit_tangents(centerline_xy: np.ndarray) -> np.ndarray:
        d = np.diff(centerline_xy, axis=0, append=centerline_xy[:1])
        n = np.linalg.norm(d, axis=1, keepdims=True)
        n = np.where(n < 1e-12, 1.0, n)
        return d / n

    def _compute_unit_left_normals(tangents: np.ndarray) -> np.ndarray:
        return np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)

    def _arc_length(centerline_xy: np.ndarray) -> np.ndarray:
        seg = np.linalg.norm(np.diff(centerline_xy, axis=0, append=centerline_xy[:1]), axis=1)
        s = np.cumsum(seg)
        s = np.insert(s, 0, 0.0)
        return s  # length M+1, last = total length

    def _wrap_s(s: np.ndarray, L: float) -> np.ndarray:
        return np.mod(s, L)

    def _frenet_to_xy_batch(centerline_xy: np.ndarray,
                            s_ref: np.ndarray,
                            sn_batch: np.ndarray) -> np.ndarray:
        """
        Convert (s,n) to XY along a closed track defined by centerline_xy and cumulative arc s_ref.
        sn_batch: (K,2) with columns [s, n]
        """
        N = centerline_xy.shape[0]
        L = s_ref[-1]
        s = _wrap_s(sn_batch[:, 0], L)
        n = sn_batch[:, 1]

        # segment index so that s in [s_ref[i], s_ref[i+1])
        idx = np.searchsorted(s_ref, s, side='right') - 1
        idx = np.clip(idx, 0, N - 1)
        idx_next = (idx + 1) % N

        s_i = s_ref[idx]
        ds_i = s_ref[idx_next] - s_ref[idx]
        ds_i = np.where(ds_i <= 0.0, ds_i + L, ds_i)  # wrap at end

        alpha = np.where(ds_i > 1e-12, (s - s_i) / ds_i, 0.0)
        p_i = centerline_xy[idx]
        p_j = centerline_xy[idx_next]
        p = p_i + (p_j - p_i) * alpha[:, None]

        tang = _compute_unit_tangents(centerline_xy)
        left_n = _compute_unit_left_normals(tang)
        n_vec = left_n[idx]

        xy = p + n[:, None] * n_vec
        return xy

    # ----------------- checks & geometry -----------------
    assert ref_xy.ndim == 2 and ref_xy.shape[1] == 2, "ref_xy must be (M,2)"
    track_left = np.asarray(track_left).reshape(-1)
    track_right = np.asarray(track_right).reshape(-1)
    assert track_left.shape[0] == ref_xy.shape[0], "track_left length must match ref_xy"
    assert track_right.shape[0] == ref_xy.shape[0], "track_right length must match ref_xy"

    s_ref = _arc_length(ref_xy)                            # (M+1,)
    tang = _compute_unit_tangents(ref_xy)
    left_n = _compute_unit_left_normals(tang)
    left_b = ref_xy + left_n * track_left[:, None]
    right_b = ref_xy - left_n * track_right[:, None]

    is_s = _get_idx(idx_s)
    is_n = _get_idx(idx_n)

    sn_mpc = X_mpc[:, [is_s, is_n]]
    sn_nom = X_nom[:, [is_s, is_n]]
    xy_mpc = _frenet_to_xy_batch(ref_xy, s_ref, sn_mpc)
    xy_nom = _frenet_to_xy_batch(ref_xy, s_ref, sn_nom)
    xy_corr = None
    if X_nomres is not None:
        sn_corr = X_nomres[:, [is_s, is_n]]
        xy_corr = _frenet_to_xy_batch(ref_xy, s_ref, sn_corr)

    # ----------------- figure layout -----------------
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.1, 1.0], hspace=0.35)

    # Left: map
    ax_map = fig.add_subplot(gs[:, 0])
    ax_map.plot(ref_xy[:, 0], ref_xy[:, 1], lw=2, label="Centerline")
    ax_map.plot(left_b[:, 0], left_b[:, 1], ls="--", label="Left boundary")
    ax_map.plot(right_b[:, 0], right_b[:, 1], ls="--", label="Right boundary")
    ax_map.plot(xy_mpc[:, 0], xy_mpc[:, 1], ls=":", label="MPC prediction traj")
    ax_map.plot(xy_nom[:, 0], xy_nom[:, 1], lw=2, label="Nominal traj")
    if xy_corr is not None:
        ax_map.plot(xy_corr[:, 0], xy_corr[:, 1], ls="-.", label="Nominal+residual traj")

    ax_map.set_title((titles or {}).get("left_title", "Track & Trajectories"))
    ax_map.set_xlabel("x [m]")
    ax_map.set_ylabel("y [m]")
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.grid(True, alpha=0.3)
    ax_map.legend(loc="best")

    # Right: 3 time series
    axs = [fig.add_subplot(gs[i, 1]) for i in range(3)]
    K = X_mpc.shape[0]
    t = np.arange(K)

    for i, name in enumerate(names):
        j = _get_idx(name)
        # actual first (thick), then nominal, then corrected if any
        axs[i].plot(t, X_mpc[:, j], label="prediction", linewidth=2)
        axs[i].plot(t, X_nom[:, j], label="nominal", linestyle="--")
        if X_nomres is not None:
            axs[i].plot(t, X_nomres[:, j], label="nominal+residual", linestyle="-.")

        axs[i].set_ylabel(name)
        axs[i].grid(True, alpha=0.3)
        if i == 0:
            axs[i].legend(loc="best")

    axs[-1].set_xlabel("step")
    fig.suptitle((titles or {}).get("suptitle", "Closed-loop (Nominal vs Actual vs Corrected)"))
    fig.tight_layout()
    return fig, (ax_map, axs)
