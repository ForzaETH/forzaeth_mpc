# readwrite_global_waypoints_no_ros.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Any, Dict
import json
import os

# ---- Minimal ROS-like message shims (only what's needed) ----

@dataclass
class Wpnt:
    id: int = 0
    s_m: float = 0.0
    d_m: float = 0.0
    x_m: float = 0.0
    y_m: float = 0.0
    d_right: float = 0.0
    d_left: float = 0.0
    psi_rad: float = 0.0
    kappa_radpm: float = 0.0
    vx_mps: float = 0.0
    ax_mps2: float = 0.0

@dataclass
class WpntArray:
    wpnts: List[Wpnt]

# Optional stubs to mirror your NamedTuple API (string/float only)
@dataclass
class String:
    data: str

@dataclass
class Float32:
    data: float

# We won’t rebuild Marker/MarkerArray fully unless you need them downstream.
# If you only use centerline waypoints and maybe est_lap_time/map_info, this is enough.

@dataclass
class GlobalWaypointsLite:
    map_info_str: String
    est_lap_time: Float32
    centerline_waypoints: WpntArray
    # The rest are optional; include if you actually consume them:
    global_traj_wpnts_iqp: Optional[WpntArray] = None
    global_traj_wpnts_sp: Optional[WpntArray] = None
    global_traj_wpnts_mintime: Optional[WpntArray] = None
    # If you need markers later, add lightweight equivalents.

# ---- Helpers to be tolerant with JSON shapes produced by rospy_message_converter ----

def _val(x: Any) -> Any:
    """Unwraps {'data': v} to v; otherwise returns x."""
    return x.get("data") if isinstance(x, dict) and "data" in x else x

def _parse_wpnt(d: Dict[str, Any]) -> Wpnt:
    # Be tolerant to missing keys by defaulting to zero
    return Wpnt(
        id=int(d.get("id", 0)),
        s_m=float(d.get("s_m", 0.0)),
        d_m=float(d.get("d_m", 0.0)),
        x_m=float(d.get("x_m", 0.0)),
        y_m=float(d.get("y_m", 0.0)),
        d_right=float(d.get("d_right", 0.0)),
        d_left=float(d.get("d_left", 0.0)),
        psi_rad=float(d.get("psi_rad", 0.0)),
        kappa_radpm=float(d.get("kappa_radpm", 0.0)),
        vx_mps=float(d.get("vx_mps", 0.0)),
        ax_mps2=float(d.get("ax_mps2", 0.0)),
    )

def _parse_wpnt_array(obj: Any) -> WpntArray:
    """
    Accepts a few shapes:
      - {'wpnts': [ {id:.., x_m:.., ...}, ... ]}
      - [ {id:.., x_m:.., ...}, ... ]
      - {'data': {'wpnts': [...]} }  (overly nested)
    """
    # unwrap {'data': ...}
    obj = _val(obj)

    if isinstance(obj, dict) and "wpnts" in obj:
        items = obj["wpnts"]
    elif isinstance(obj, list):
        items = obj
    elif isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], dict) and "wpnts" in obj["data"]:
        items = obj["data"]["wpnts"]
    else:
        # Fallback: empty
        items = []

    return WpntArray([_parse_wpnt(it) for it in items])

# ---- Public API (no ROS required) ----

def read_global_waypoints(map_name: str, maps_root: str) -> GlobalWaypointsLite:
    """
    Read the JSON written by the original ROS-based writer, but without ROS.
    You must pass the absolute path to the maps folder (i.e., the parent of map_name).

    Example:
        maps_root = "/home/gil/F1TENTH/race_stack/stack_master/maps"
        gw = read_global_waypoints("ETHZ_A_floor", maps_root)
    """
    path = os.path.join(maps_root, map_name, "global_waypoints.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"global_waypoints.json not found at: {path}")

    with open(path, "r") as f:
        d: Dict[str, Any] = json.load(f)

    map_info_str = String(str(_val(d.get("map_info_str", ""))))
    est_lap_time = Float32(float(_val(d.get("est_lap_time", 0.0))))

    # Required
    centerline_waypoints = _parse_wpnt_array(d.get("centerline_waypoints", []))

    # Optional, only parse if present
    global_traj_wpnts_iqp = _parse_wpnt_array(d["global_traj_wpnts_iqp"]) if "global_traj_wpnts_iqp" in d else None
    global_traj_wpnts_sp = _parse_wpnt_array(d["global_traj_wpnts_sp"]) if "global_traj_wpnts_sp" in d else None
    global_traj_wpnts_mintime = _parse_wpnt_array(d["global_traj_wpnts_mintime"]) if "global_traj_wpnts_mintime" in d else None

    return GlobalWaypointsLite(
        map_info_str=map_info_str,
        est_lap_time=est_lap_time,
        centerline_waypoints=centerline_waypoints,
        global_traj_wpnts_iqp=global_traj_wpnts_iqp,
        global_traj_wpnts_sp=global_traj_wpnts_sp,
        global_traj_wpnts_mintime=global_traj_wpnts_mintime,
    )
