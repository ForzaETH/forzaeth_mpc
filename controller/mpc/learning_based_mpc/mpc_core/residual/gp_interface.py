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
    Build 3 independent GPs (one per output dimension: vx, vy, r).
    Each has its own kernel hyperparameters (lengthscale, outputscale, noise).
    """
    _torch_guard(device)
    import gpytorch
    import torch

    assert Z.ndim == 2 and Y.ndim == 2, "Z:(N,Din), Y:(N,Dout)"
    x_train = torch.tensor(Z, dtype=torch.float32, device=device)
    y_train = torch.tensor(Y, dtype=torch.float32, device=device)
    nout = Y.shape[1]
    assert nout == 3, "Expected 3 outputs: vx, vy, r"

    class IndependentGP(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=Z.shape[1])
            )

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    # Create a list of (model, likelihood) pairs — one per output dimension
    models = []
    likelihoods = []
    for i in range(nout):
        yi = y_train[:, i]
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        model = IndependentGP(x_train, yi, likelihood).to(device)
        models.append(model)
        likelihoods.append(likelihood)

    return models, likelihoods


def train_gp_model(models, likelihoods, torch_seed=456, training_iterations=200, lr=0.03, verbose=True):
    import gpytorch
    import torch
    torch.manual_seed(torch_seed)

    trained_models = []
    trained_likelihoods = []

    for i, (gp_model, likelihood) in enumerate(zip(models, likelihoods)):
        gp_model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(gp_model.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)

        train_x = gp_model.train_inputs[0]
        train_y = gp_model.train_targets

        for it in range(training_iterations):
            optimizer.zero_grad()
            output = gp_model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
            if verbose and (it % max(1, training_iterations // 5) == 0 or it == training_iterations - 1):
                print(f"[GP {i}] iter {it:4d}/{training_iterations}: loss={loss.item():.6f}")

        gp_model.eval()
        likelihood.eval()
        trained_models.append(gp_model)
        trained_likelihoods.append(likelihood)

    return trained_models, trained_likelihoods


def gp_predictor_from_model(models, likelihoods, device="cpu"):
    import torch
    for m, l in zip(models, likelihoods):
        m.eval()
        l.eval()

    def predict_mean(Zstar: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            zt = torch.tensor(Zstar, dtype=torch.float32, device=device)
            preds = []
            for m, l in zip(models, likelihoods):
                pred = l(m(zt))
                preds.append(pred.mean.detach().cpu().numpy().reshape(-1, 1))
            return np.hstack(preds)
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
