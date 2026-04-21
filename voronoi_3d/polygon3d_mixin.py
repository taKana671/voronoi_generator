import numpy as np
from ..polygon_mixin import PolygonMixin


class Polygon3DMixin(PolygonMixin):

    def sort_3d_vertices_ccw(self, vertices):
        """
        Sorts 3D vertices in counter-clockwise order.
         Args:
            vertices (numpy.ndarray): The vertices of a 3D polygon. 

         Returns:
            numpy.ndarray: The sorted vertices.
        """
        center = np.mean(vertices, axis=0)

        # Determine the normal vector of the plane using SVD or cross product.
        # SVD is more robust for approximate planes.
        P = vertices - center
        _, _, V = np.linalg.svd(P)
        # The normal vector is the third column of V (least significant singular value direction)
        normal = V[2, :]

        # Project to a local 2D plane.
        ref_vec = [0., 1., 0.] if abs(normal[0]) > abs(normal[1]) else [1., 0., 0.]
        u = self.cross(normal, ref_vec)

        norm = (u[0] ** 2 + u[1] ** 2 + u[2] ** 2) ** 0.5
        u /= norm
        v = self.cross(normal, u)

        # Project the centered points onto the (u, v) plane
        projected_2d = np.array([[np.dot(p, u), np.dot(p, v)] for p in P])

        # Sort in 2D by angle.
        angles = np.arctan2(projected_2d[:, 1], projected_2d[:, 0])
        sorted_indices = np.argsort(angles)

        # Reorder the original 3D vertices.
        sorted_vertices = vertices[sorted_indices]

        return sorted_vertices

    def cross(self, a, b):
        # Using numpy.cross caused performance issues, so use the Python implementation
        # of cross instead (It is because numpy.cross is used on a small numpy.ndarray with shape=(3, ) ?).
        return np.array([
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]
        ])