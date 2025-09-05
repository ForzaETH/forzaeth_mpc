from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

def plot_centerline(centerline) -> None:
    """Plot the centerline and add an arrow indicating the direction."""
    direction = (centerline[1][0] - centerline[0][0], centerline[1][1] - centerline[0][1])
    direction_magnitude = (direction[0]**2 + direction[1]**2)**0.5
    direction = (direction[0] / direction_magnitude, direction[1] / direction_magnitude)

    x, y = zip(*centerline)
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='Centerline', color='blue')

    # Add arrow at the first waypoint
    plt.arrow(x[0], y[0], direction[0], direction[1],
              head_width=0.5, head_length=0.5, fc='red', ec='red', label='Direction')

    plt.title('Centerline with Direction Arrow')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_centerline_with_lr(centerline_xy: Tuple[Tuple], left_dist: Tuple[float], right_dist: Tuple[float]) -> None:
    """Plot the centerline along with left and right distances, adding an arrow for direction."""

    # extract directions for centerline
    centerline_direction = (centerline_xy[1:] - centerline_xy[:-1])
    last_dir = centerline_xy[0] - centerline_xy[-1]
    centerline_direction = np.vstack((centerline_direction, last_dir)) # repeat last direction for same length
    centerline_direction = centerline_direction / np.linalg.norm(centerline_direction, axis=1)[:, None]  # normalize to unit vectors

    centerline_direction *= 1
    assert centerline_xy.shape == centerline_direction.shape, "shapes of centerline and direction must match"

    # compute left and right boundary points
    left_boundary = centerline_xy + np.vstack((-centerline_direction[:, 1], centerline_direction[:, 0])).T * left_dist[:, None]
    right_boundary = centerline_xy + np.vstack((centerline_direction[:, 1], -centerline_direction[:, 0])).T * right_dist[:, None]

    # plot everything
    plt.figure(figsize=(10, 6))
    plt.plot(centerline_xy[:, 0], centerline_xy[:, 1], label='Centerline', color='blue')
    plt.plot(left_boundary[:, 0], left_boundary[:, 1], label='Left Boundary', color='green')
    plt.plot(right_boundary[:, 0], right_boundary[:, 1], label='Right Boundary', color='orange')
    plt.arrow(centerline_xy[0, 0], centerline_xy[0, 1],
              centerline_direction[0, 0], centerline_direction[0, 1],
              head_width=0.5, head_length=0.5, fc='red', ec='red', label='Direction')
    plt.title('Centerline with Left and Right Boundaries')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.show()

