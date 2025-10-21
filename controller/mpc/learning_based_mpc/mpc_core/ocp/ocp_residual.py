# ocp_residual.py
from acados_template import AcadosOcp
import l4acados as l4a
def build_residual_mpc(ocp: AcadosOcp, residual_model: l4a.PyTorchResidualModel) -> l4a.ResidualLearningMPC:
    return l4a.ResidualLearningMPC(ocp=ocp, residual_model=residual_model)
