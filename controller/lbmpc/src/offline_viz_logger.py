import csv
import glob
import os
import time

import numpy as np


class OfflineMPCLogger:
    def __init__(self, root_dir, nx, nu, N, state_names, input_names,
                 enable=True, flush_every=25, horizon_stride=1):
        self.enable = bool(enable)
        self.flush_every = int(flush_every)
        self.horizon_stride = int(horizon_stride)

        self.nx, self.nu, self.N = int(nx), int(nu), int(N)
        self.state_names = list(state_names)
        self.input_names = list(input_names)

        self.step = 0
        self.t0_wall = time.time()
        self.t0_perf = time.perf_counter()

        self.base_dir = os.path.join(root_dir, "data", "offline_viz")
        self.csv_dir = os.path.join(self.base_dir, "csv")
        self.fig_dir = os.path.join(self.base_dir, "figs")
        os.makedirs(self.csv_dir, exist_ok=True)
        os.makedirs(self.fig_dir, exist_ok=True)

        # delete old csvs + figures at start
        for pat in [os.path.join(self.csv_dir, "*.csv"),
                    os.path.join(self.fig_dir, "*.png")]:
            for f in glob.glob(pat):
                try:
                    os.remove(f)
                except Exception:
                    pass

        self._open_files()
        self._rows_since_flush = 0

    def _open_files(self):
        self.f_step = open(os.path.join(self.csv_dir, "step_log.csv"), "w", newline="")
        self.f_hp = open(os.path.join(self.csv_dir, "horizon_pred.csv"), "w", newline="")
        self.f_hb = open(os.path.join(self.csv_dir, "horizon_bounds.csv"), "w", newline="")
        self.f_hc = open(os.path.join(self.csv_dir, "horizon_covdiag.csv"), "w", newline="")

        self.w_step = csv.writer(self.f_step)
        self.w_hp = csv.writer(self.f_hp)
        self.w_hb = csv.writer(self.f_hb)
        self.w_hc = csv.writer(self.f_hc)

        # Write headers if files are empty
        if self.f_step.tell() == 0:
            step_cols = (
                ["t_wall", "t_rel", "step", "lap", "status"]
                + [f"x_meas_{n}" for n in self.state_names]
                + [f"x_prop_{n}" for n in self.state_names]
                + [f"x_pred_delay_{n}" for n in self.state_names]
                + [f"x_nom_pred_{n}" for n in self.state_names]
                + [f"u0_{n}" for n in self.input_names]
                + ["t_total_ms", "t_init_ms", "t_residual_ms", "t_rest_ms"]
                + ["r_meas_0", "r_meas_1", "r_meas_2", "r_pred_0", "r_pred_1", "r_pred_2"]
                + ["gp_num_points"]
            )
            self.w_step.writerow(step_cols)

        if self.f_hp.tell() == 0:
            self.w_hp.writerow(
                ["t_rel", "step", "lap", "node"]
                + [f"x_{n}" for n in self.state_names]
                + [f"u_{n}" for n in self.input_names]
            )

        if self.f_hb.tell() == 0:
            # bounds length is unknown at init; we’ll store dynamic-length vectors as fixed max = 16
            # (safe; extra columns stay blank if shorter).
            max_h = 16
            self.max_h = max_h
            self.w_hb.writerow(
                ["t_rel", "step", "lap", "node"]
                + [f"lh_{i}" for i in range(max_h)]
                + [f"uh_{i}" for i in range(max_h)]
                + ["lh_t0", "uh_t0", "lh_t1", "uh_t1"]
            )

        if self.f_hc.tell() == 0:
            self.w_hc.writerow(
                ["t_rel", "step", "lap", "node"]
                + [f"Pdiag_{n}" for n in self.state_names]
            )

    def close(self):
        for f in [self.f_step, self.f_hp, self.f_hb, self.f_hc]:
            try:
                f.flush()
                f.close()
            except Exception:
                pass

    def _flush_if_needed(self):
        self._rows_since_flush += 1
        if self._rows_since_flush >= self.flush_every:
            for f in [self.f_step, self.f_hp, self.f_hb, self.f_hc]:
                try:
                    f.flush()
                except Exception:
                    pass
            self._rows_since_flush = 0

    # ------------------- robust getters -------------------
    @staticmethod
    def _try_get(obj, stage, field):
        if obj is None:
            return None
        # l4a solver style
        try:
            return obj.get(stage, field)
        except Exception:
            pass
        # acados style
        try:
            return obj.constraints_get(stage, field)
        except Exception:
            pass
        return None

    def log_step(self, lap, status,
                 x_meas, x_prop, x_pred_delay, x_nom_pred,
                 u0,
                 t_total, t_init, t_residual, t_rest,
                 r_meas, r_pred,
                 gp_num_points):

        if not self.enable:
            return

        t_wall = time.time()
        t_rel = time.perf_counter() - self.t0_perf

        x_meas = np.asarray(x_meas, dtype=float).reshape(-1)
        x_prop = np.asarray(x_prop, dtype=float).reshape(-1)
        x_pred_delay = np.asarray(x_pred_delay, dtype=float).reshape(-1)
        x_nom_pred = np.asarray(x_nom_pred, dtype=float).reshape(-1)
        u0 = np.asarray(u0, dtype=float).reshape(-1)

        # residuals (store first 3)
        rm = np.full(3, np.nan)
        rp = np.full(3, np.nan)
        if r_meas is not None:
            rr = np.asarray(r_meas, dtype=float).reshape(-1)
            rm[: min(3, rr.size)] = rr[: min(3, rr.size)]
        if r_pred is not None:
            rr = np.asarray(r_pred, dtype=float).reshape(-1)
            rp[: min(3, rr.size)] = rr[: min(3, rr.size)]

        row = (
            [t_wall, t_rel, self.step, int(lap), int(status)]
            + x_meas.tolist()
            + x_prop.tolist()
            + x_pred_delay.tolist()
            + x_nom_pred.tolist()
            + u0.tolist()
            + [1e3 * t_total, 1e3 * t_init, 1e3 * t_residual, 1e3 * t_rest]
            + rm.tolist() + rp.tolist()
            + [gp_num_points if gp_num_points is not None else ""]
        )
        self.w_step.writerow(row)
        self._flush_if_needed()

    def log_horizon(self, solver, lap, covariances_array=None, idx_lh_t=None, idx_uh_t=None):
        if not self.enable:
            return
        if (self.step % self.horizon_stride) != 0:
            return

        t_rel = time.perf_counter() - self.t0_perf

        ocp_solver = getattr(solver, "ocp_solver", None)
        idx_lh_t = idx_lh_t or []
        idx_uh_t = idx_uh_t or []

        for k in range(self.N + 1):
            xk = self._try_get(solver, k, "x")
            if xk is None:
                xk = self._try_get(ocp_solver, k, "x")
            if xk is None:
                continue
            xk = np.asarray(xk, dtype=float).reshape(-1)

            if k < self.N:
                uk = self._try_get(solver, k, "u")
                if uk is None:
                    uk = self._try_get(ocp_solver, k, "u")
                uk = np.asarray(uk, dtype=float).reshape(-1) if uk is not None else np.full(self.nu, np.nan)
            else:
                uk = np.full(self.nu, np.nan)

            self.w_hp.writerow([t_rel, self.step, int(lap), k] + xk.tolist() + uk.tolist())

            # --- bounds (nonlinear constraints): lh / uh ---
            lh = self._try_get(ocp_solver, k, "lh")
            uh = self._try_get(ocp_solver, k, "uh")
            lh = np.asarray(lh, dtype=float).reshape(-1) if lh is not None else np.array([])
            uh = np.asarray(uh, dtype=float).reshape(-1) if uh is not None else np.array([])

            # pad to max_h
            max_h = getattr(self, "max_h", 16)
            lh_pad = np.full(max_h, np.nan)
            uh_pad = np.full(max_h, np.nan)
            lh_pad[: min(max_h, lh.size)] = lh[: min(max_h, lh.size)]
            uh_pad[: min(max_h, uh.size)] = uh[: min(max_h, uh.size)]

            # tightened indices (first two are enough for quick plots)
            def safe_idx(arr, idx):
                return float(arr[idx]) if (idx is not None and idx < arr.size) else np.nan

            lh_t0 = safe_idx(lh, idx_lh_t[0]) if len(idx_lh_t) > 0 else np.nan
            uh_t0 = safe_idx(uh, idx_uh_t[0]) if len(idx_uh_t) > 0 else np.nan
            lh_t1 = safe_idx(lh, idx_lh_t[1]) if len(idx_lh_t) > 1 else np.nan
            uh_t1 = safe_idx(uh, idx_uh_t[1]) if len(idx_uh_t) > 1 else np.nan

            self.w_hb.writerow([t_rel, self.step, int(lap), k] + lh_pad.tolist() +
                               uh_pad.tolist() + [lh_t0, uh_t0, lh_t1, uh_t1])

            # --- covariance diag from covariances_array (ZoRO) ---
            if covariances_array is not None:
                nx = self.nx
                start = (nx * nx) * k
                Pflat = np.asarray(covariances_array[start: start + nx * nx], dtype=float)
                if Pflat.size == nx * nx:
                    P = Pflat.reshape((nx, nx), order="F")  # col-major
                    self.w_hc.writerow([t_rel, self.step, int(lap), k] + np.diag(P).tolist())

        self._flush_if_needed()

    def tick(self):
        self.step += 1
