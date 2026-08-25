"""A live view of a running simulation.

The static figure that `--plot` writes shows where the car went. This one shows what the
controller is thinking while it drives: the predicted horizon, which moves every step and
which no end-of-run picture can convey.

It needs an interactive matplotlib backend. Without one it disables itself and says so,
rather than stopping the run.
"""
from __future__ import annotations

import numpy as np

S, N, VX = 0, 1, 3  # state indices, the same for every model here


class LiveView:
    """One figure, updated in place. Call `update` from the simulation loop."""

    def __init__(self, waypoints: np.ndarray, track_len: float, title: str,
                 history_s: float = 6.0):
        self.ok = False
        self.track_len = float(track_len)
        self.waypoints = waypoints
        self.history_s = float(history_s)

        import matplotlib
        # ask matplotlib which backends can show a window. Do not test the name: every
        # interactive backend here is called something like QtAgg or TkAgg, so a check for
        # an "agg" suffix rejects all of them.
        backend = matplotlib.get_backend()
        interactive = {b.lower() for b in matplotlib.rcsetup.interactive_bk}
        if backend.lower() not in interactive:
            print(f"[live] backend {backend!r} cannot show a window; live view disabled. "
                  "Unset MPLBACKEND, or use --plot for the summary.")
            return

        import matplotlib.pyplot as plt
        self.plt = plt
        self.fig, (self.ax_xy, self.ax_v) = plt.subplots(
            1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.15, 1]})
        self.fig.suptitle(title)

        self.ax_xy.plot(waypoints[:, 0], waypoints[:, 1], "k--", lw=0.6, label="raceline")
        # the driven path, the predicted horizon, and the car itself
        self.path, = self.ax_xy.plot([], [], lw=1.0, color="tab:blue", label="driven")
        self.horizon, = self.ax_xy.plot([], [], lw=2.0, color="tab:orange",
                                        label="predicted horizon")
        self.car, = self.ax_xy.plot([], [], "o", ms=7, color="tab:red", label="car")
        self.ax_xy.set_aspect("equal")
        self.ax_xy.legend(loc="upper right", fontsize=8)
        self.ax_xy.set_xlabel("x [m]"), self.ax_xy.set_ylabel("y [m]")

        self.speed, = self.ax_v.plot([], [], lw=1.0, color="tab:blue")
        self.ax_v.set_xlabel("t [s]"), self.ax_v.set_ylabel("$v_x$ [m/s]")
        self.ax_v.grid(alpha=0.3)

        self.fig.tight_layout()
        self.fig.show()
        self.ok = True

    def update(self, X: np.ndarray, sd: np.ndarray, dt: float, laps: int = 0) -> None:
        """`X` is the state history, `sd` the predicted (s, n) over the horizon."""
        if not self.ok or X.size == 0:
            return
        from forzaeth_mpc.sim.cli.run_laps import frenet_to_xy

        keep = max(1, int(self.history_s / dt))
        tail = X[-keep:]
        px, py = frenet_to_xy(self.waypoints, tail[:, S], tail[:, N], self.track_len)
        car_xy = (px[-1:], py[-1:])
        px, py = self._break_at_lap_wrap(tail[:, S], px, py)
        self.path.set_data(px, py)
        self.car.set_data(*car_xy)

        if sd is not None and len(sd):
            sd = np.asarray(sd, dtype=float)
            hx, hy = frenet_to_xy(self.waypoints, sd[:, 0], sd[:, 1], self.track_len)
            self.horizon.set_data(*self._break_at_lap_wrap(sd[:, 0], hx, hy))

        t = np.arange(X.shape[0]) * dt
        self.speed.set_data(t[-keep:], tail[:, VX])
        self.ax_v.relim(), self.ax_v.autoscale_view()
        self.ax_xy.relim(), self.ax_xy.autoscale_view()

        self.ax_v.set_title(f"lap {laps} | t = {t[-1]:.1f} s | "
                            f"$v_x$ = {tail[-1, VX]:.2f} m/s | n = {tail[-1, N]:+.3f} m")
        try:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except Exception as exc:            # the user closed the window
            print(f"[live] window gone ({type(exc).__name__}); continuing without it")
            self.ok = False

    def _break_at_lap_wrap(self, s: np.ndarray, x: np.ndarray, y: np.ndarray):
        """Insert a gap where s crosses the start line.

        The path is drawn from (s, n), so a lap boundary is a jump of nearly a whole track
        length. Without a break, the line closes that jump with a chord straight across the
        track.
        """
        jump = np.abs(np.diff(np.mod(s, self.track_len))) > 0.5 * self.track_len
        if not jump.any():
            return x, y
        cut = np.flatnonzero(jump) + 1
        return (np.insert(x, cut, np.nan), np.insert(y, cut, np.nan))

    def save(self, path: str) -> None:
        if self.ok:
            self.fig.savefig(path, dpi=110)

    def close(self) -> None:
        if self.ok:
            self.plt.close(self.fig)
            self.ok = False
