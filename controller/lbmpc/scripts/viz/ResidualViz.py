# residual_viz.py
import os
import time
from collections import deque
from typing import Optional, Sequence

import matplotlib

# Ensure a GUI backend (Qt or Tk) — do this BEFORE importing pyplot.
if os.environ.get("MPLBACKEND", "") == "":
    try:
        matplotlib.use("QtAgg")   # preferred on Linux
    except Exception:
        matplotlib.use("TkAgg")   # fallback
import matplotlib.pyplot as plt
import numpy as np


class ResidualRealtimeViz:
    """
    Real-time rolling visualization for residuals (vx, vy, gz).
    Shows measured residual and predicted residual (no confidence band).

    Use:
        viz = ResidualRealtimeViz(window_seconds=10.0, target_hz=30.0)
        ...
        viz.update(t, pred, measured)

    Where:
        - t: float (seconds, monotonic; if None, uses time.monotonic()).
        - pred:     array-like, shape (3,) for (vx, vy, gz) predicted residuals.
        - measured: array-like or None, shape (3,) measured residuals.
    """

    LABELS = ("vx", "vy", "gz")

    def __init__(self,
                 window_seconds: float = 12.0,
                 target_hz: float = 30.0,
                 figure_title: str = "Residuals (measured vs predicted)"):
        self.window_s = float(window_seconds)
        self.target_dt = 1.0 / float(target_hz)

        # Rolling buffers
        self.t = deque()
        self.meas = [deque(), deque(), deque()]   # measured residuals per channel
        self.pred = [deque(), deque(), deque()]   # predicted residuals per channel

        # Figure/axes
        self.fig, self.axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        try:
            self.fig.canvas.manager.set_window_title(figure_title)
        except Exception:
            pass
        self.fig.suptitle(figure_title)

        self.lines_meas = []
        self.lines_pred = []

        for i, ax in enumerate(self.axes):
            ax.set_ylabel(f"{self.LABELS[i]} residual")
            ax.grid(True, linestyle=":", linewidth=0.6)
            # Initialize lines with empty data
            (line_meas,) = ax.plot([], [], lw=1.5, label="measured error")
            (line_pred,) = ax.plot([], [], lw=1.5, label="predicted error")
            self.lines_meas.append(line_meas)
            self.lines_pred.append(line_pred)
            ax.legend(loc="upper left")

        self.axes[-1].set_xlabel("time [s]")
        self._last_draw = 0.0
        plt.ion()
        plt.show(block=False)

    def _trim_window(self):
        """Trim buffers to the rolling time window."""
        if not self.t:
            return
        t_min = self.t[-1] - self.window_s

        while self.t and (self.t[0] < t_min):
            self.t.popleft()
            for i in range(3):
                if self.meas[i]:
                    self.meas[i].popleft()
                if self.pred[i]:
                    self.pred[i].popleft()

    def update(self,
               t: Optional[float],
               pred: Sequence[float],
               measured: Optional[Sequence[float]] = None):
        """
        Append one sample and refresh the plot if needed.
        """
        if t is None:
            t = time.monotonic()

        # Append
        self.t.append(float(t))
        for i in range(3):
            self.pred[i].append(float(pred[i]))
            if measured is not None:
                self.meas[i].append(float(measured[i]))
            else:
                self.meas[i].append(np.nan)

        # Trim to window
        self._trim_window()

        # Prepare numpy arrays for fast set_data
        x = np.asarray(self.t, dtype=float)
        for i, ax in enumerate(self.axes):
            y_pred = np.asarray(self.pred[i], dtype=float)
            y_meas = np.asarray(self.meas[i], dtype=float)

            # Update lines
            self.lines_pred[i].set_data(x, y_pred)
            self.lines_meas[i].set_data(x, y_meas)

            # Keep y-lims sensible
            y_all = np.concatenate([y_pred, np.nan_to_num(y_meas, nan=0.0)])
            if y_all.size > 0:
                y_min = np.nanmin(y_all)
                y_max = np.nanmax(y_all)
                if np.isfinite(y_min) and np.isfinite(y_max):
                    pad = 1.0 if y_min == y_max else 0.1 * (y_max - y_min)
                    ax.set_ylim(y_min - pad, y_max + pad)

            # Rolling x-lim
            if x.size:
                ax.set_xlim(x[0], x[-1])

        # Throttle redraws to target rate
        now = time.monotonic()
        if now - self._last_draw >= self.target_dt:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            plt.pause(0.001)
            self._last_draw = now
