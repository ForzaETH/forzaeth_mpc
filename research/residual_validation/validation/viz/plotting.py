# plotting.py
from typing import Optional, Union

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import torch
from forzaeth_mpc.utils.indices import StateIndex

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


def plot_gp_training_fit(Z, Y, gp_models, likelihoods, device="cpu", titles=("vx", "vy", "wz")):
    """
    Plot GP training fits for multiple independent GPs (one per output: vx, vy, wz).

    Parameters
    ----------
    Z : (N, din) np.ndarray
        GP inputs used for training or evaluation.
    Y : (N, m) np.ndarray
        Observed targets (residuals). For your case, m = 3 → (vx, vy, wz).
    gp_models : list[gpytorch.models.ExactGP]
        List of trained GP models (length = m).
    likelihoods : list[gpytorch.likelihoods.Likelihood]
        Corresponding list of likelihoods (length = m).
    device : str
        "cpu" or "cuda".
    titles : tuple[str]
        Names for each output dimension; defaults to ("vx", "vy", "wz").

    Returns
    -------
    fig, axes : matplotlib Figure and Axes
    """
    # Sanity check
    m = len(gp_models)
    assert m == len(likelihoods), "Number of models and likelihoods must match"
    assert Y.shape[1] == m, f"Y has {Y.shape[1]} outputs but {m} GPs provided"

    Z = np.asarray(Z, dtype=float)
    Y = np.asarray(Y, dtype=float)
    N = Y.shape[0]
    t = np.arange(N)

    fig, axes = plt.subplots(m, 1, figsize=(9, 5 + 2 * (m - 1)), sharex=True)
    if m == 1:
        axes = [axes]

    # Loop over each GP output
    for j, (gp_model, likelihood) in enumerate(zip(gp_models, likelihoods)):
        gp_model.eval()
        likelihood.eval()

        # Align Z with GP input dimension (usually 3)
        try:
            din_gp = int(gp_model.train_inputs[0].shape[-1])
        except Exception:
            din_gp = Z.shape[1]
        if Z.shape[1] != din_gp:
            Z_eval = Z[:, :din_gp]
        else:
            Z_eval = Z

        # Evaluate posterior
        with torch.no_grad():
            Z_t = torch.tensor(Z_eval, dtype=torch.float32, device=device)
            post = likelihood(gp_model(Z_t))
            mu = post.mean.detach().cpu().numpy().reshape(-1)
            std = post.variance.sqrt().detach().cpu().numpy().reshape(-1)

        ax = axes[j]
        ax.scatter(t, Y[:, j], s=10, alpha=0.5, label="Observed Data")
        ax.plot(t, mu, lw=2, label="Mean")
        ax.fill_between(t, mu - 2 * std, mu + 2 * std, alpha=0.25, label="Confidence")

        title = titles[j] if j < len(titles) else f"output {j}"
        ax.set_title(f"GP fit for residual: {title}")
        ax.grid(True, ls=":", alpha=0.6)
        if j == m - 1:
            ax.set_xlabel("sample index")
        ax.set_ylabel("residual")
        ax.legend(loc="best")

    plt.tight_layout()
    return fig, axes


# --- NEW: residual surfaces for vy, wz as f(vy, wz) ---


def plot_residual_surfaces_vy_wz(
    Z: np.ndarray,
    Y: np.ndarray,
    *,
    titles=("Residual vy ~ (vy, wz)", "Residual wz ~ (vy, wz)"),
    alpha=0.9,
    elev=25,
    azim=-135,
):
    """
    Make two 3D triangulated surfaces:
      z1 = residual_vy(vy, wz) and z2 = residual_wz(vy, wz),
    using the GP training data (or any Z, Y you pass in).

    Assumes Z[:, 1] = vy, Z[:, 2] = wz
            Y[:, 1] = residual vy, Y[:, 2] = residual wz
    """
    Z = np.asarray(Z, dtype=float)
    Y = np.asarray(Y, dtype=float)
    assert Z.ndim == 2 and Y.ndim == 2, "Z and Y must be 2D"
    assert Z.shape[0] == Y.shape[0], "Z and Y must have same N"
    assert Z.shape[1] >= 3 and Y.shape[1] >= 3, "Need at least 3 cols (vx, vy, wz)"

    vy = Z[:, 1].reshape(-1)
    wz = Z[:, 2].reshape(-1)
    r_vy = Y[:, 1].reshape(-1)   # residual vy
    r_wz = Y[:, 2].reshape(-1)   # residual wz

    tri = mtri.Triangulation(vy, wz)

    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(1, 2, wspace=0.25)

    # --- Left: residual vy surface ---
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    s1 = ax1.plot_trisurf(tri, r_vy, linewidth=0.1, antialiased=True, alpha=alpha)
    ax1.set_title(titles[0])
    ax1.set_xlabel("vy input")
    ax1.set_ylabel("wz input")
    ax1.set_zlabel("residual vy")
    ax1.view_init(elev=elev, azim=azim)
    fig.colorbar(s1, ax=ax1, shrink=0.7, pad=0.1)

    # --- Right: residual wz surface ---
    ax2 = fig.add_subplot(gs[0, 1], projection="3d")
    s2 = ax2.plot_trisurf(tri, r_wz, linewidth=0.1, antialiased=True, alpha=alpha)
    ax2.set_title(titles[1])
    ax2.set_xlabel("vy input")
    ax2.set_ylabel("wz input")
    ax2.set_zlabel("residual wz")
    ax2.view_init(elev=elev, azim=azim)
    fig.colorbar(s2, ax=ax2, shrink=0.7, pad=0.1)

    fig.suptitle("Residual dependence on (vy, wz)")
    fig.tight_layout()
    return fig, (ax1, ax2)


def plot_residual_surfaces_vy_wz_colored_by_vx(
    Z: np.ndarray,
    Y: np.ndarray,
    *,
    titles=("Residual vy ~ (vy, wz)", "Residual wz ~ (vy, wz)"),
    cmap_name="viridis",
    alpha=0.95,
    elev=25,
    azim=-135,
):
    """
    3D trisurfaces: residual_vy(vy, wz) and residual_wz(vy, wz),
    with surface color driven by vx (per-triangle mean vx).

    Axes:
        X = vy
        Y = wz
        Z = residual (vy or wz)
        color = vx (triangle-mean)
    """
    Z = np.asarray(Z, dtype=float)
    Y = np.asarray(Y, dtype=float)

    assert Z.ndim == 2 and Y.ndim == 2, "Z and Y must be 2D"
    assert Z.shape[0] == Y.shape[0], "Z and Y must have same N"
    assert Z.shape[1] >= 3 and Y.shape[1] >= 3, "Expect columns [vx, vy, wz]"

    vx = Z[:, 0]
    vy = Z[:, 1]
    wz = Z[:, 2]
    r_vy = Y[:, 1]
    r_wz = Y[:, 2]

    # Triangulate on (vy, wz)
    tri = mtri.Triangulation(vy, wz)

    # Compute a scalar per-triangle color = mean vx of the triangle
    tri_ix = tri.triangles  # (ntri, 3)
    vx_tri_mean = np.mean(vx[tri_ix], axis=1)

    # Colormap setup for colorbar
    norm = mcolors.Normalize(vmin=float(np.min(vx)), vmax=float(np.max(vx)))
    cmap = cm.get_cmap(cmap_name)

    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(1, 2, wspace=0.28)

    # --- Left: residual vy surface ---
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    s1 = ax1.plot_trisurf(
        tri,
        r_vy,
        linewidth=0.05,
        antialiased=True,
        shade=False,       # important to prevent shading from overriding colors
        alpha=alpha,
    )
    # attach colors *after* creating the surface
    s1.set_cmap(cmap)
    s1.set_norm(norm)
    s1.set_array(vx_tri_mean)     # per-triangle scalar
    s1.autoscale_None()           # ensure color limits apply

    ax1.set_title(titles[0])
    ax1.set_xlabel("vy input")
    ax1.set_ylabel("wz input")
    ax1.set_zlabel("residual vy")
    ax1.view_init(elev=elev, azim=azim)

    # --- Right: residual wz surface ---
    ax2 = fig.add_subplot(gs[0, 1], projection="3d")
    s2 = ax2.plot_trisurf(
        tri,
        r_wz,
        linewidth=0.05,
        antialiased=True,
        shade=False,
        alpha=alpha,
    )
    s2.set_cmap(cmap)
    s2.set_norm(norm)
    s2.set_array(vx_tri_mean)
    s2.autoscale_None()

    ax2.set_title(titles[1])
    ax2.set_xlabel("vy input")
    ax2.set_ylabel("wz input")
    ax2.set_zlabel("residual wz")
    ax2.view_init(elev=elev, azim=azim)

    # One shared colorbar for vx
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # required for matplotlib < 3.8
    cbar = fig.colorbar(sm, ax=[ax1, ax2], shrink=0.75, pad=0.08)
    cbar.set_label("vx (triangle-mean)")

    fig.suptitle("Residual dependence on (vy, wz) with vx coloring")
    fig.tight_layout()
    return fig, (ax1, ax2)


def plot_dict_kept_dropped(Z_kept, Z_dropped, dims=None, title="Dictionary events", save_path=None, three_d=False):
    """
    Overlay plot of kept vs dropped points.
    dims: tuple of indices to plot (i,j[,k]); defaults to first two dims.
    three_d: set True to use 3D when dims has length 3.
    """
    if isinstance(Z_kept, list) and len(Z_kept) > 0:
        Zk = np.vstack(Z_kept)
    elif isinstance(Z_kept, np.ndarray):
        Zk = Z_kept
    else:
        Zk = np.empty((0, 3))
    if isinstance(Z_dropped, list) and len(Z_dropped) > 0:
        Zd = np.vstack(Z_dropped)
    elif isinstance(Z_dropped, np.ndarray):
        Zd = Z_dropped
    else:
        Zd = np.empty((0, 3))

    # pick dims
    nz = max(Zk.shape[1] if Zk.size else 0, Zd.shape[1] if Zd.size else 0)
    if dims is None:
        if nz < 3:
            raise ValueError(f"Need at least 3 dims to plot 3D, got nz={nz}. Provide dims=(i,j,k).")
        dims = (0, 1, 2)
    if len(dims) != 3:
        raise ValueError(f"dims must be a tuple of 3 indices, got {dims}")

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')

    if Zk.size:
        ax.scatter(Zk[:, dims[0]], Zk[:, dims[1]], Zk[:, dims[2]],
                   s=18, alpha=0.9, label="kept", depthshade=True)
    if Zd.size:
        ax.scatter(Zd[:, dims[0]], Zd[:, dims[1]], Zd[:, dims[2]],
                   s=20, alpha=0.9, marker="x", label="dropped", depthshade=False)

    ax.set_xlabel(f"z[{dims[0]}]")
    ax.set_ylabel(f"z[{dims[1]}]")
    ax.set_zlabel(f"z[{dims[2]}]")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
        plt.close(fig)
        print(f"Saved: {save_path}")
    else:
        plt.show()
