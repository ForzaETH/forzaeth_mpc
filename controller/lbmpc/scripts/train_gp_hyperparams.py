#!/usr/bin/env python3
import gpytorch
import numpy as np
import torch
from l4acados.models.pytorch_models.gpytorch_models.gpytorch_gp import \
    BatchIndependentMultitaskGPModel


def main():
    # Load data
    z = np.loadtxt("../data/datapoints/z_data.csv", delimiter=",")
    y = np.loadtxt("../data/datapoints/y_data.csv", delimiter=",")

    # According to feature selection for the GP
    gp_input_mask = np.array([0, 0, 0, 1, 1, 1, 1, 1, 0, 0], dtype=bool)
    z = z[:, gp_input_mask]

    z = torch.as_tensor(z, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32)
    if y.ndim == 1:
        y = y.unsqueeze(-1)

    n_tasks = y.shape[1]

    # Define likelihood and GP
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=n_tasks)
    gp_model = BatchIndependentMultitaskGPModel(
        train_x=z,
        train_y=y,
        likelihood=likelihood,
        use_ard=True,
    )

    # Train hyperparameters
    gp_model, likelihood = train_gp_model(gp_model, torch_seed=0)

    # Extract and print hyperparameters
    with torch.no_grad():
        task_noises = likelihood.task_noises.detach().cpu().numpy().tolist()
        lengthscale = (
            gp_model.covar_module.base_kernel.lengthscale
            .squeeze()
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )
        outputscale = (
            gp_model.covar_module.outputscale
            .squeeze()
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )

    print("hyperparameters:")
    print(f"  likelihood.task_noises: {task_noises}")
    print("  kernel.lengthscale:")
    for row in lengthscale:
        print(f"    - {row}")
    print(f"  kernel.outputscale: {outputscale}")


def train_gp_model(
    gp_model, torch_seed=None, training_iterations=400, learning_rate=0.1
):
    if torch_seed is not None:
        torch.manual_seed(torch_seed)

    likelihood = gp_model.likelihood
    train_x = gp_model.train_inputs[0]
    train_y = gp_model.train_targets

    # Find optimal model hyperparameters
    gp_model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, gp_model.parameters()), lr=learning_rate
    )  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)

    prev_loss = np.inf
    num_loss_below_threshold = 0

    for i in range(training_iterations):
        optimizer.zero_grad()
        output = likelihood(gp_model(train_x))
        loss = -mll(output, train_y.reshape((train_y.numel(),)))

        num_loss_below_threshold = (
            num_loss_below_threshold + 1 if abs(loss - prev_loss) < 5e-4 else 0
        )

        if num_loss_below_threshold > 5:
            print(f"stopping GP optimization early after {i} iterations.")
            break

        prev_loss = loss

        loss.backward()
        if (i + 1) % 20 == 0:
            print("Iter %d/%d - Loss: %.3f" % (i + 1, training_iterations, loss.item()))

        optimizer.step()

    gp_model.eval()
    likelihood.eval()
    return gp_model, likelihood


if __name__ == "__main__":
    main()
