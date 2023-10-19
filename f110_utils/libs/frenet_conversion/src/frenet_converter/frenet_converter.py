from typing import Union
import numpy as np
from scipy.interpolate import CubicSpline

class FrenetConverter:
    def __init__(self, waypoints_x: np.array, waypoints_y: np.array):
        self.waypoints_x = waypoints_x
        self.waypoints_y = waypoints_y
        self.waypoints_s = None
        self.spline_x = None
        self.spline_y = None
        self.raceline_length = None
        self.waypoints_distance_m = 0.1 # [m]
        self.iter_max = 3

        self.build_raceline()

    def build_raceline(self):
        self.waypoints_s = [0.0]
        prev_wpnt_x =  self.waypoints_x[0]
        prev_wpnt_y =  self.waypoints_y[0]
        for wpnt_x, wpnt_y in zip(self.waypoints_x[1:], self.waypoints_y[1:]):
            dist = np.linalg.norm([wpnt_x - prev_wpnt_x, wpnt_y - prev_wpnt_y])
            prev_wpnt_x = wpnt_x
            prev_wpnt_y = wpnt_y
            self.waypoints_s.append(self.waypoints_s[-1] + dist)        
        self.spline_x = CubicSpline(self.waypoints_s, self.waypoints_x)
        self.spline_y = CubicSpline(self.waypoints_s, self.waypoints_y)
        self.raceline_length = self.waypoints_s[-1]

    def get_frenet(self, x, y, s=None) -> np.array:
        # Compute Frenet coordinates for a given (x, y) point
        if s is None:
            s = self.get_approx_s(x, y)
            s, d = self.get_frenet_coord(x, y, s)
        else:
            s, d = self.get_frenet_coord(x, y, s)

        # print(f"OG s and d {s, d}")

        return np.array([s, d])

    def get_approx_s(self, x, y) -> float:
        """
        Finds the s-coordinate of the given point by finding the nearest waypoint.
        """
        # get distance with broadcasting multiple arrays
        lenx = len(x)
        dist_x = x - np.tile(self.waypoints_x, (lenx, 1)).T
        dist_y = y - np.tile(self.waypoints_y, (lenx, 1)).T
        # print(dist_x.shape, dist_x)
        return np.argmin(np.linalg.norm([dist_x.T, dist_y.T], axis=0), axis=1)*self.waypoints_distance_m

    def get_frenet_coord(self, x, y, s, eps_m=0.01) -> float:
        """
        Finds the s-coordinate of the given point, considering the perpendicular
        projection of the point on the track.
        
        Args:
            x (float): x-coordinate of the point
            y (float): y-coordinate of the point
            s (float): estimated s-coordinate of the point
            eps_m (float): maximum error tolerance for the projection. Default is 0.01.
        
        Returns:
            The s-coordinate of the point on the track.
        """
        # Check if point is on the estimated s perpendicular to the track

        is_perpendicular, projection, d = self.check_perpendicular(x, y, s, eps_m)
        print(f"projection {projection}, the d {d}, the s {s}")
        # if np.abs(projection) > eps_m:
        is_perpendicular = True
        for i in range(self.iter_max):
            cand_s = (s + projection)%self.raceline_length
            is_perpendicular, cand_projection, cand_d = self.check_perpendicular(x, y, cand_s, eps_m)
            cand_projection = np.clip(cand_projection, -self.waypoints_distance_m/(2*self.iter_max), self.waypoints_distance_m/(2*self.iter_max))
            updated_idxs = np.abs(cand_projection) < np.abs(projection)
            d[updated_idxs] = cand_d[updated_idxs]
            s[updated_idxs] = cand_s[updated_idxs]
            projection[updated_idxs] = cand_projection[updated_idxs]

        return s, d

    def check_perpendicular(self, x, y, s, eps_m=0.01) -> Union[bool, float]:
        # obtain unit vector parallel to the track
        dx_ds, dy_ds = self.get_derivative(s)
        tangent = np.array([dx_ds, dy_ds])
        if np.any(np.isnan(s)):
            raise ValueError("BUB FRENET CONVERTER: S is nan")
        tangent /= np.linalg.norm(tangent, axis=0)
        # print(f"tangents {tangent[:, :5]}")

        # obtain vector from the track to the point
        x_vec = x - self.spline_x(s)
        y_vec = y - self.spline_y(s)
        point_to_track = np.array([x_vec, y_vec])
        # print("PTT", point_to_track.shape, point_to_track[:, :5])
        # print(f"shape xvec{x_vec.shape}, yvec{y_vec.shape}, sshape{s.shape}")
        # print(f"tang, {tangent}, dx_s {dx_ds}, dys {dy_ds}, sx{self.spline_x(s)}, sy {self.spline_y(s)}")
        
        # check if the vectors are perpendicular
        # the dot product computes the projected component of point_to_track onto tangent in meters
        # proj = np.dot(tangent, point_to_track)
        # print(tangent.shape, point_to_track.shape)
        # print(tangent)
        # print(x,y)
        # print(self.spline_x(s), self.spline_y(s))
        # print(point_to_track)
        proj = np.einsum('ij,ij->j', tangent, point_to_track)
        # print(f"proj {proj[:5]}")
        perps = np.array([-tangent[1, :], tangent[0, :]])
        # print(f"perps: {perps[:, :5]}")
        d = np.einsum('ij,ij->j', perps, point_to_track)
        # print(f"d {d[:5]}")
        return None, proj, d

        # if (proj > 0 and proj < eps_m) or (proj < 0 and proj > -eps_m):
        #     # compute d here, for efficiency
        #     d = np.dot([-tangent[1], tangent[0]], point_to_track)

        #     return True, proj, d
        # else:
        #     d = np.dot([-tangent[1], tangent[0]], point_to_track)
        #     return False, proj, d
    
    def get_derivative(self, s) -> np.array:
        """
        Returns the derivative of the point corresponding to s on the chosen line. 
        
        Args: 
            s: parameter which is used to evaluate the spline
            line: argument used to choose the line. Can be 'int', 'mid', 'out'. Default is 'mid'.

        Returns:
            der: dx/ds, dy/ds
        """
        s = s%self.raceline_length

        der = [self.spline_x(s, 1), self.spline_y(s, 1)]
        
        return der
    

    def get_cartesian(self, s: float, d: float) -> np.array:
        """
        Convert Frenet coordinates to Cartesian coordinates
        
        Args:
            s (float): longitudinal coordinate
            d (float): lateral coordinate
            
        Returns:
            np.array: [x, y] Cartesian coordinates
        """
        x = self.spline_x(s)
        y = self.spline_y(s)
        psi = self.get_derivative(s)
        psi = np.arctan2(psi[1], psi[0])
        x += d * np.cos(psi + np.pi / 2)
        y += d * np.sin(psi + np.pi / 2)
        
        return np.array([x, y])
    


