# residual_models.py
from __future__ import annotations


def make_residual_model_from_gp(gp_model, likelihood=None):
    # Match inverted-pendulum usage: just wrap the trained GP into l4acados' residual model
    from learning_based_mpc.l4acados.src.l4acados.models import \
        GPyTorchResidualModel

    # Put models in eval mode (safe even if already eval)
    try:
        gp_model.eval()
    except Exception:
        pass
    if likelihood is not None:
        try:
            likelihood.eval()
            # if your gp_model expects the likelihood attached, set it:
            if not hasattr(gp_model, "likelihood"):
                gp_model.likelihood = likelihood
        except Exception:
            pass
    return GPyTorchResidualModel(gp_model)
