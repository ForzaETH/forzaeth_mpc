#!/usr/bin/env bash
set -euo pipefail

echo "=== [install_l4acados] start ==="

# ---- Configuration (safe defaults; override by exporting env before calling) ----
# Repo root on the running container (your repo is mounted here by default)
PROJECT_DIR="${PROJECT_DIR:-/home/${USER:-$(whoami)}/catkin_ws/src/race_stack}"

# Submodule path
L4A_REL="controller/mpc/learning_based_mpc/l4acados"
L4A_DIR="${PROJECT_DIR}/${L4A_REL}"
L4A_EXTERNAL_DIR="${L4A_DIR}/external"

# ACADOS location (*already installed* earlier via install_acados.sh ~/acados)
ACADOS_SOURCE_DIR="${ACADOS_SOURCE_DIR:-/home/${USER:-$(whoami)}/acados}"

# Respect your CI/EDITABLE toggle (same meaning as in post_installation.sh)
EDITABLE="${EDITABLE:-true}"

# Idempotency marker (keyed by current l4acados commit so it re-runs if you update the submodule)
L4A_COMMIT="$(git -C "${L4A_DIR}" rev-parse --verify HEAD 2>/dev/null || echo unknown)"
MARKER_DIR="/home/${USER:-$(whoami)}/.cache"
MARKER_FILE="${MARKER_DIR}/.l4acados_installed_${L4A_COMMIT}"

# ---- Preconditions ----
if [ ! -d "${L4A_DIR}" ]; then
  echo "[install_l4acados] ${L4A_DIR} not found; skipping."
  exit 0
fi

mkdir -p "${MARKER_DIR}"
if [ -f "${MARKER_FILE}" ]; then
  echo "[install_l4acados] already installed for commit ${L4A_COMMIT}; nothing to do."
  exit 0
fi

# ---- Ensure Python tooling ----
/usr/bin/python3 -m pip install --no-cache-dir --upgrade pip
# Installing Cython removes the need for any ~/.local/bin/cython shim
/usr/bin/python3 -m pip install --no-cache-dir "Cython>=0.29"

# ---- If ACADOS exists, install its Python template & persist user env ----
if [ -d "${ACADOS_SOURCE_DIR}" ]; then
  if [ -d "${ACADOS_SOURCE_DIR}/interfaces/acados_template" ]; then
    echo "[install_l4acados] Installing acados_template from ${ACADOS_SOURCE_DIR} ..."
    /usr/bin/python3 -m pip install --no-cache-dir -e "${ACADOS_SOURCE_DIR}/interfaces/acados_template"
  else
    echo "[install_l4acados] WARNING: ${ACADOS_SOURCE_DIR}/interfaces/acados_template missing; skipping."
  fi

  # Add env vars to user's ~/.bashrc (local; no /etc writes)
  if ! grep -q "ACADOS_SOURCE_DIR=" ~/.bashrc 2>/dev/null; then
    {
      echo "export ACADOS_SOURCE_DIR=\"${ACADOS_SOURCE_DIR}\""
      echo "export LD_LIBRARY_PATH=\"\${ACADOS_SOURCE_DIR}/lib:\${LD_LIBRARY_PATH:-}\""
    } >> ~/.bashrc
    echo "[install_l4acados] Appended ACADOS env to ~/.bashrc"
  fi
else
  echo "[install_l4acados] WARNING: ACADOS not found at ${ACADOS_SOURCE_DIR}; continuing without acados_template."
fi

# ---- Make l4acados/external importable without PYTHONPATH (write a .pth) ----
/usr/bin/python3 - <<'PY' "${L4A_EXTERNAL_DIR}"
import site, pathlib, sys

ext = pathlib.Path(sys.argv[1]).resolve()

# Prefer user site-packages to avoid permission issues
candidates = [site.getusersitepackages()]

# Also try global site-packages (may be read-only for the current user)
try:
    candidates.extend(site.getsitepackages())
except Exception:
    pass  # not always available

wrote = False
for sp in candidates:
    try:
        sp_path = pathlib.Path(sp)
        sp_path.mkdir(parents=True, exist_ok=True)
        pth = sp_path / 'l4acados_external.pth'
        pth.write_text(str(ext) + '\n')
        print(f"[install_l4acados] wrote {pth} -> {ext}")
        wrote = True
        break
    except (PermissionError, OSError) as e:
        print(f"[install_l4acados] cannot write to {sp}: {e}")

if not wrote:
    sys.exit("[install_l4acados] ERROR: failed to write l4acados_external.pth anywhere")
PY

# ---- Install l4acados (editable or not; with gpytorch extras) ----
if [ "${EDITABLE}" = false ] ; then
  echo "[install_l4acados] pip install ${L4A_DIR}[gpytorch]"
  /usr/bin/python3 -m pip install --no-cache-dir "${L4A_DIR}[gpytorch]"
else
  echo "[install_l4acados] pip install -e ${L4A_DIR}[gpytorch]"
  /usr/bin/python3 -m pip install --no-cache-dir -e "${L4A_DIR}[gpytorch]"
fi

# ---- Done ----
touch "${MARKER_FILE}"
echo "=== [install_l4acados] done for commit ${L4A_COMMIT} ==="
