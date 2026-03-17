"""
mpc_sim.cli — command-line entry points (runnable scripts) for the mpc_sim package.

Run any script directly with the -m switch from the repo root, e.g.:

    python3 -m mpc_sim.cli.run_single_rollout
    python3 -m mpc_sim.cli.analyze_cost_vs_mu
    python3 -m mpc_sim.cli.tune_mpc_weights
"""

from .analyze_cost_vs_mu import main as analyze_cost_vs_mu
from .noise_robustness_analysis import main as noise_robustness_analysis
from .run_single_rollout import main as run_single_rollout
from .tune_mpc_weights import main as tune_mpc_weights

__all__ = [
    "run_single_rollout",
    "analyze_cost_vs_mu",
    "tune_mpc_weights",
    "noise_robustness_analysis"]
