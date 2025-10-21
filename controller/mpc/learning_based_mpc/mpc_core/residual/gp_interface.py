# gp_interface.py
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np


def _torch_guard(device: str = "cpu"):
    try:
        import gpytorch  # noqa: F401
        import torch  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "gp_interface requires torch and gpytorch installed and importable."
        ) from e
    return None


def build_multitask_gp(Z: np.ndarray, Y: np.ndarray, device: str = "cpu"):
    """
    Create a Multitask GP like the owners' example:
      - MultitaskGaussianLikelihood(num_tasks = Y.shape[1])
      - BatchIndependentMultitask-style ExactGP model
    Shapes:
      Z: (N, Din), Y: (N, Dout)
    """
    _torch_guard(device)
    import gpytorch
    import torch

    assert Z.ndim == 2 and Y.ndim == 2, "Z:(N,Din), Y:(N,Dout)"
    x_train = torch.tensor(Z, dtype=torch.float32, device=device)
    y_train = torch.tensor(Y, dtype=torch.float32, device=device)
    nout = Y.shape[1]

    class BatchIndependentMultitaskGPModel(gpytorch.models.ExactGP):
        """
        Minimal re-implementation consistent with the owners' naming.
        We model D outputs jointly via a MultitaskGaussianLikelihood and an ICM kernel,
        but keep it simple/robust for residual learning.
        """

        def __init__(self, train_x, train_y, likelihood):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.MultitaskMean(
                gpytorch.means.ConstantMean(), num_tasks=nout
            )
            # Use an ICM (Intrinsic Coregionalization) kernel on an RBF base
            base_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
            self.covar_module = gpytorch.kernels.MultitaskKernel(
                base_kernel, num_tasks=nout, rank=min(2, nout)
            )

        def forward(self, x):
            # x: (N, Din)
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=nout).to(device)
    gp_model = BatchIndependentMultitaskGPModel(x_train, y_train, likelihood).to(device)
    return gp_model, likelihood


def train_gp_model(gp_model, likelihood, torch_seed: int = 456,
                   training_iterations: int = 200, lr: float = 0.03, verbose: bool = True):
    """
    Train exactly like the owners' example: maximize marginal log-likelihood (exact GP).
    """
    import gpytorch
    import torch
    torch.manual_seed(torch_seed)

    gp_model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(gp_model.parameters(), lr=lr)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)

    train_x = gp_model.train_inputs[0]  # from ExactGP base
    train_y = gp_model.train_targets    # (N, Dout)

    for it in range(training_iterations):
        optimizer.zero_grad()
        output = gp_model(train_x)  # MultitaskMultivariateNormal
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        if verbose and (it % max(1, training_iterations // 5) == 0 or it == training_iterations - 1):
            print(f"[GP] iter {it:4d}/{training_iterations}: loss={loss.item():.6f}")

    gp_model.eval()
    likelihood.eval()
    return gp_model, likelihood


def gp_predictor_from_model(gp_model, likelihood, device: str = "cpu") -> Callable[[np.ndarray], np.ndarray]:
    """
    Return a numpy->numpy callable predict_mean(Z*) consistent with your previous code.
    """
    import torch
    gp_model.eval()
    likelihood.eval()

    def predict_mean(Zstar: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            zt = torch.tensor(Zstar, dtype=torch.float32, device=device)
            pred = likelihood(gp_model(zt))
            # Multitask: pred.mean shape (N, Dout)
            return pred.mean.detach().cpu().numpy()
    return predict_mean


def save_gp_state(gp_model, likelihood, model_path: str, lik_path: str) -> None:
    import torch
    torch.save(gp_model.state_dict(), model_path)
    torch.save(likelihood.state_dict(), lik_path)


def load_gp_state(gp_model, likelihood, model_path: str, lik_path: str) -> None:
    import torch
    gp_model.load_state_dict(torch.load(model_path, map_location=next(gp_model.parameters()).device))
    likelihood.load_state_dict(torch.load(lik_path, map_location=next(likelihood.parameters()).device))
    gp_model.eval()
    likelihood.eval()
