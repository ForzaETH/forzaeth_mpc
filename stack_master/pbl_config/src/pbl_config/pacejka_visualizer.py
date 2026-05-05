#!/usr/bin/env python3
"""
Minimal Pacejka tire visualizer - no ROS required
Plots lateral tire force curves for front and rear tires
"""
import fire
import numpy as np
import matplotlib.pyplot as plt
from pbl_config import load_pacejka_tire_config_ros


def pacejka_formula(B, C, D, E, alpha, F_z):
    """
    Pacejka tire model formula
    Args:
        B, C, D, E: Pacejka coefficients
        alpha: slip angle (rad)
        F_z: vertical load (N)
    Returns:
        Lateral force (N)
    """
    y = F_z * D * np.sin(C * np.arctan(B * alpha - E * (B * alpha - np.arctan(B * alpha))))
    return y


def plot_pacejka_curves(config, racecar_version, floor, F_z=30, alpha_range=None):
    """
    Plot Pacejka curves for front and rear tires
    Args:
        config: dict with tire parameters
        racecar_version: name of the racecar
        floor: floor/track name
        F_z: vertical load in Newtons
        alpha_range: slip angle range in degrees (default: -40 to 40)
    """
    if alpha_range is None:
        alpha_range = np.linspace(-40, 40, 300)
    else:
        alpha_range = np.linspace(alpha_range[0], alpha_range[1], 300)

    # Convert degrees to radians
    alpha_rad = np.deg2rad(alpha_range)

    # Front tire
    F_front = pacejka_formula(
        config['Bf'], config['Cf'], config['Df'], config['Ef'],
        alpha_rad, F_z
    )

    # Rear tire
    F_rear = pacejka_formula(
        config['Br'], config['Cr'], config['Dr'], config['Er'],
        alpha_rad, F_z
    )

    # Plot overlay
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.plot(alpha_range, F_front, 'b-', linewidth=2.5, label='Front tire')
    ax.plot(alpha_range, F_rear, 'r-', linewidth=2.5, label='Rear tire')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Slip Angle (degrees)', fontsize=12)
    ax.set_ylabel('Lateral Force (N)', fontsize=12)
    ax.set_title(f'{racecar_version} - Pacejka Tire Model (F_z={F_z}N, {floor})',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)

    # Print parameters
    print(f"\n{'='*60}")
    print(f"Racecar: {racecar_version} | Floor: {floor}")
    print(f"{'='*60}")
    print(f"Front Tire (B={config['Bf']:.4f}, C={config['Cf']:.4f}, D={config['Df']:.4f}, E={config['Ef']:.4f})")
    print(f"Rear Tire  (B={config['Br']:.4f}, C={config['Cr']:.4f}, D={config['Dr']:.4f}, E={config['Er']:.4f})")
    print(f"Friction Coefficient: {config['friction_coeff']}")
    max_front_idx = np.argmax(np.abs(F_front))
    max_rear_idx = np.argmax(np.abs(F_rear))
    print(f"Peak lateral forces at {alpha_range[max_front_idx]:.1f}° (front), {alpha_range[max_rear_idx]:.1f}° (rear)")
    print(f"{'='*60}\n")

    plt.tight_layout()
    plt.show()


def main(racecar_version: str = "NUC5", floor: str = "dubi"):
    """
    Visualize Pacejka tire curves

    Args:
        racecar_version: Racecar version name (default: NUC5)
        floor: Floor/track name (default: dubi)
    """
    print(f"Loading Pacejka config: {racecar_version} / {floor}")
    config = load_pacejka_tire_config_ros(racecar_version, floor)

    config_dict = {
        'Bf': config.Bf,
        'Cf': config.Cf,
        'Df': config.Df,
        'Ef': config.Ef,
        'Br': config.Br,
        'Cr': config.Cr,
        'Dr': config.Dr,
        'Er': config.Er,
        'friction_coeff': config.friction_coeff,
        'floor': config.floor
    }

    plot_pacejka_curves(config_dict, racecar_version, floor, F_z=3000)


if __name__ == "__main__":
    fire.Fire(main)
