import os
from pathlib import Path

import yaml

_ENV_VAR = 'FORZAETH_MPC_CONFIG'

# repo_root/forzaeth_mpc/config/paths.py -> repo_root/config
_DEFAULT_CONFIG_ROOT = Path(__file__).resolve().parents[2] / 'config'


def config_root() -> Path:
    """Directory holding the per-car config dirs (NUC5/, DEFAULT/, ...).
    Override with the FORZAETH_MPC_CONFIG env var; on the car point it at
    $(rospack find stack_master)/config."""
    override = os.environ.get(_ENV_VAR)
    return Path(override) if override else _DEFAULT_CONFIG_ROOT


class StrictLoader(yaml.SafeLoader):
    """SafeLoader that rejects a repeated mapping key instead of taking the last one.

    Plain ``yaml.safe_load`` is last-wins and silent. That is how
    ``correct_v_y_dot`` came to be declared twice in NUC5's and NUC8's
    ``single_track_mpc_params.yaml`` -- the file reads ``true`` on one line and
    ``false`` nine lines later, so STMPC has been running a different lateral
    acceleration in its tire constraint than the config appears to say. Pydantic
    cannot catch it: by the time the schema sees the dict, one of the two values
    is simply gone.
    """

    def construct_mapping(self, node, deep=False):
        seen = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise yaml.YAMLError(
                    f"duplicate key {key!r} in {key_node.start_mark}"
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def strict_safe_load(stream):
    """``yaml.safe_load``, except a repeated key is an error, not last-wins."""
    return yaml.load(stream, Loader=StrictLoader)


if __name__ == "__main__":
    import io

    clean = "a: 1\nb: 2\nnested:\n  c: 3\n"
    assert strict_safe_load(io.StringIO(clean)) == {"a": 1, "b": 2, "nested": {"c": 3}}

    for bad in ("a: 1\nb: 2\na: 3\n",            # top level, the NUC5 shape
                "nested:\n  c: 3\n  c: 4\n"):    # nested mapping
        try:
            strict_safe_load(io.StringIO(bad))
        except yaml.YAMLError as exc:
            assert "duplicate key" in str(exc), exc
        else:
            raise AssertionError(f"duplicate key not rejected: {bad!r}")

    print(f"OK: strict_safe_load rejects duplicate keys; config_root() = {config_root()}")
