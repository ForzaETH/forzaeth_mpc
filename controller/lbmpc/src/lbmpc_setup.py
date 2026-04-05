import os
import re
from dataclasses import dataclass
from glob import glob
from typing import Optional

import gpytorch
import torch
from l4acados.models import GPyTorchResidualModel, PyTorchFeatureSelector
from l4acados.models.pytorch_models.gpytorch_models.gpytorch_data_processing_strategy import (
    OnlineLearningStrategy,
    VoidDataStrategy,
)
from l4acados.models.pytorch_models.gpytorch_models.gpytorch_gp import (
    BatchIndependentInducingPointGPModel,
    BatchIndependentMultitaskGPModel,
)
from lbmpc.src.offline_viz_logger import OfflineMPCLogger
from lbmpc.src.utils.indicies import InputIndex, StateIndex
from lbmpc.src.utils.utils import (
    check_file_changes,
    load_gp_model_bundle,
    save_gp_model_bundle,
)


@dataclass
class SolverBuildConfig:
    build_c_code: bool
    path_json_ocp: str
    path_json_sim: str


@dataclass
class ResidualModelSetup:
    residual_gp: GPyTorchResidualModel
    input_feature_selector: PyTorchFeatureSelector
    use_inducing_gp: bool
    min_points_before_using_gp: int
    max_num_data_points: int
    nx_gp: int
    nu_gp: int
    mode: str
    learnt_gp_model_path: str
    gp_warmup_done: bool


def build_solver_build_config(lbmpc_path: str, params_path: str) -> SolverBuildConfig:
    solver_settings_file_name = os.path.join(lbmpc_path, "src", "acados_solver_settings.py")
    bicycle_model_file_name = os.path.join(lbmpc_path, "src", "bicycle_model.py")
    params_file_name = os.path.join(params_path, "lb_mpc_params.yaml")
    mpc_files_to_watch = [
        solver_settings_file_name,
        bicycle_model_file_name,
        params_file_name,
    ]

    build_hash_path = os.path.join(lbmpc_path, "data/hash", "build_c_code_hash.md5")
    path_json_ocp = os.path.join(lbmpc_path, "data/ocp", "lbmpc_ocp_solver_config.json")
    path_json_sim = os.path.join(lbmpc_path, "data/ocp", "lbmpc_sim_solver_config.json")

    build_c_code = check_file_changes(mpc_files_to_watch, build_hash_path)
    build_c_code = build_c_code or not cython_solver_artifacts_available(
        path_json_ocp=path_json_ocp,
        path_json_sim=path_json_sim,
    )

    return SolverBuildConfig(
        build_c_code=build_c_code,
        path_json_ocp=path_json_ocp,
        path_json_sim=path_json_sim,
    )


def cython_solver_artifacts_available(path_json_ocp: str, path_json_sim: str) -> bool:
    for json_path, module_stem in (
        (path_json_ocp, "acados_ocp_solver_pyx"),
        (path_json_sim, "acados_sim_solver_pyx"),
    ):
        code_export_directory = read_code_export_directory(json_path)
        if not code_export_directory:
            return False

        artifact_pattern = os.path.join(code_export_directory, f"{module_stem}*.so")
        if not glob(artifact_pattern):
            return False

    return True


def read_code_export_directory(json_path: str) -> Optional[str]:
    if not os.path.exists(json_path):
        return None

    try:
        with open(json_path, "r", encoding="utf-8") as json_file:
            match = re.search(
                r'"code_export_directory"\s*:\s*"([^"]+)"',
                json_file.read(),
            )
    except OSError:
        return None

    if not match:
        return None

    return os.path.abspath(match.group(1))


def build_offline_logger(
    lbmpc_path: str,
    ocp,
    horizon: int,
    horizon_requested: bool,
    use_zoro: bool,
    enabled: bool,
):
    if horizon_requested and not use_zoro:
        print("[offline_viz] Horizon logging requested, but use_zoro is False. Disabling horizon logging.")

    if not enabled:
        return None

    try:
        nx = int(ocp.dims.nx)
        nu = int(ocp.dims.nu)
        state_names = [e.name.lower() for e in sorted(StateIndex, key=lambda x: x.value)]
        input_names = [e.name.lower() for e in sorted(InputIndex, key=lambda x: x.value)]

        return OfflineMPCLogger(
            root_dir=lbmpc_path,
            nx=nx,
            nu=nu,
            N=int(horizon),
            state_names=state_names,
            input_names=input_names,
            enable=True,
            flush_every=25,
            horizon_stride=4,
        )
    except Exception as exc:
        print(f"[offline_viz] logger disabled: {exc}")
        return None


def initialize_residual_model(
    model,
    gp_cfg,
    lbmpc_path: str,
    nw_gp: int,
) -> ResidualModelSetup:
    use_inducing_gp = gp_cfg.use_inducing_gp
    min_points_before_using_gp = gp_cfg.min_points_before_using_gp
    max_num_data_points = gp_cfg.max_num_data_points

    nx_gp = int(model.n_x)
    nu_gp = int(model.n_u)

    input_selection = gp_cfg.input_feature_selection
    if len(input_selection) != (nx_gp + nu_gp):
        raise ValueError(
            f"gp_params['input_feature_selection'] must have length {nx_gp + nu_gp}"
        )

    input_feature_selector = PyTorchFeatureSelector(input_selection)
    learnt_gp_model_path = os.path.join(lbmpc_path, "data", "models", "latest_gp_model.pt")

    mode = gp_cfg.data_processing_mode
    if mode == "learn":
        data_processing_strategy = OnlineLearningStrategy(max_num_data_points)
    elif mode == "use_learnt_model":
        data_processing_strategy = VoidDataStrategy()
    else:
        raise ValueError(
            f"Invalid data_processing_mode: {mode}. Must be one of ['learn', 'use_learnt_model']"
        )

    learnt_gp_bundle = None
    z_train_data, y_train_data = None, None
    learnt_num_points = None
    if mode == "use_learnt_model":
        learnt_gp_bundle = load_gp_model_bundle(learnt_gp_model_path)
        if learnt_gp_bundle is None:
            raise FileNotFoundError(
                f"[LBMPC] No learnt GP model found at {learnt_gp_model_path}. "
                "Run once with gp_params.data_processing_mode='learn' first."
            )

        z_train_data = learnt_gp_bundle.get("train_x")
        y_train_data = learnt_gp_bundle.get("train_y")
        if z_train_data is None or y_train_data is None:
            raise ValueError(
                f"[LBMPC] Learnt GP bundle at {learnt_gp_model_path} does not contain training data."
            )
        learnt_num_points = int(z_train_data.shape[0])

    input_dimension = int(sum(input_selection))
    gp_model, hyperparameter_names_dict = _build_gp_model(
        use_inducing_gp=use_inducing_gp,
        gp_cfg=gp_cfg,
        nw_gp=nw_gp,
        input_dimension=input_dimension,
        train_x=z_train_data,
        train_y=y_train_data,
    )

    hyper_cfg = gp_cfg.hyperparameters
    if learnt_gp_bundle is not None:
        _load_learnt_gp_model(
            gp_model=gp_model,
            bundle=learnt_gp_bundle,
            input_dimension=input_dimension,
            nw_gp=nw_gp,
            use_inducing_gp=use_inducing_gp,
            num_inducing_points=gp_cfg.num_inducing_points,
            model_path=learnt_gp_model_path,
        )
    elif hyper_cfg:
        gp_model.initialize(
            **{
                hyperparameter_names_dict[key]: torch.Tensor(hyper_cfg[key])
                for key in hyperparameter_names_dict.keys()
                if key in hyper_cfg
            }
        )

    gp_model.eval()
    gp_model.likelihood.eval()

    gp_warmup_done = False
    if mode == "learn":
        gp_model.train_inputs = None
        gp_model.train_targets = None
        gp_model.prediction_strategy = None
    elif mode == "use_learnt_model":
        gp_warmup_done = True
        print(
            f"[LBMPC] Loaded learnt GP model from {learnt_gp_model_path} "
            f"with {learnt_num_points} points."
        )

    residual_gp = GPyTorchResidualModel(
        gp_model=gp_model,
        feature_selector=input_feature_selector,
        data_processing_strategy=data_processing_strategy,
    )

    return ResidualModelSetup(
        residual_gp=residual_gp,
        input_feature_selector=input_feature_selector,
        use_inducing_gp=use_inducing_gp,
        min_points_before_using_gp=min_points_before_using_gp,
        max_num_data_points=max_num_data_points,
        nx_gp=nx_gp,
        nu_gp=nu_gp,
        mode=mode,
        learnt_gp_model_path=learnt_gp_model_path,
        gp_warmup_done=gp_warmup_done,
    )


def persist_learnt_gp_model(
    model_path: str,
    residual_gp,
    input_feature_selection,
    residual_dimension: int,
    use_inducing_gp: bool,
    num_inducing_points: int,
    racecar_version: str,
) -> None:
    metadata = {
        "input_dimension": int(sum(input_feature_selection)),
        "residual_dimension": int(residual_dimension),
        "use_inducing_gp": bool(use_inducing_gp),
        "num_inducing_points": int(num_inducing_points),
        "racecar_version": racecar_version,
    }

    try:
        save_gp_model_bundle(
            model_path,
            residual_gp.gp_model,
            metadata=metadata,
        )
    except Exception as exc:
        print(f"[LBMPC] Failed to save learnt GP model to {model_path}: {exc}")


def _build_gp_model(
    use_inducing_gp: bool,
    gp_cfg,
    nw_gp: int,
    input_dimension: int,
    train_x,
    train_y,
):
    if not use_inducing_gp:
        gp_model = BatchIndependentMultitaskGPModel(
            train_x=train_x,
            train_y=train_y,
            likelihood=gpytorch.likelihoods.MultitaskGaussianLikelihood(
                num_tasks=nw_gp,
            ),
            residual_dimension=nw_gp,
            input_dimension=input_dimension,
            use_ard=True,
        )
        hyperparameter_names_dict = {
            "likelihood.task_noises": "likelihood.task_noises",
            "kernel.lengthscale": "covar_module.base_kernel.lengthscale",
            "kernel.outputscale": "covar_module.outputscale",
        }
        return gp_model, hyperparameter_names_dict

    num_inducing = gp_cfg.num_inducing_points
    inducing_zeros = torch.zeros((num_inducing, input_dimension), dtype=torch.float32)
    gp_model = BatchIndependentInducingPointGPModel(
        train_x=train_x,
        train_y=train_y,
        likelihood=gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=nw_gp,
        ),
        residual_dimension=nw_gp,
        input_dimension=input_dimension,
        inducing_points=inducing_zeros,
        use_ard=True,
    )
    hyperparameter_names_dict = {
        "likelihood.task_noises": "likelihood.task_noises",
        "kernel.lengthscale": "covar_module.base_kernel.base_kernel.lengthscale",
        "kernel.outputscale": "covar_module.base_kernel.outputscale",
    }
    return gp_model, hyperparameter_names_dict


def _load_learnt_gp_model(
    gp_model,
    bundle: dict,
    input_dimension: int,
    nw_gp: int,
    use_inducing_gp: bool,
    num_inducing_points: int,
    model_path: str,
) -> None:
    metadata = bundle.get("metadata", {})

    saved_input_dim = metadata.get("input_dimension")
    if saved_input_dim is not None and int(saved_input_dim) != int(input_dimension):
        raise ValueError(
            f"[LBMPC] Learnt GP model input dimension mismatch: saved {saved_input_dim}, "
            f"current {input_dimension}."
        )

    saved_residual_dim = metadata.get("residual_dimension")
    if saved_residual_dim is not None and int(saved_residual_dim) != int(nw_gp):
        raise ValueError(
            f"[LBMPC] Learnt GP residual dimension mismatch: saved {saved_residual_dim}, "
            f"current {nw_gp}."
        )

    saved_use_inducing_gp = metadata.get("use_inducing_gp")
    if saved_use_inducing_gp is not None and bool(saved_use_inducing_gp) != bool(use_inducing_gp):
        raise ValueError(
            "[LBMPC] Learnt GP model was saved with a different use_inducing_gp setting."
        )

    saved_num_inducing_points = metadata.get("num_inducing_points")
    if (
        use_inducing_gp
        and saved_num_inducing_points is not None
        and int(saved_num_inducing_points) != int(num_inducing_points)
    ):
        raise ValueError(
            f"[LBMPC] Learnt GP model uses {saved_num_inducing_points} inducing points, "
            f"but the current config requests {num_inducing_points}."
        )

    try:
        gp_model.load_state_dict(bundle["gp_model_state_dict"])
        gp_model.likelihood.load_state_dict(bundle["likelihood_state_dict"])
    except Exception as exc:
        raise RuntimeError(
            f"[LBMPC] Failed to load learnt GP model from {model_path}: {exc}"
        ) from exc
