import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, ConvexHull, Delaunay
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree
from scipy.stats import qmc
import math

from sympy import Point3D, Plane, Polygon, Segment3D, Symbol
from sympy.geometry.entity import GeometryEntity 
import pprint


class Polygon3DMixin:

    def sort_3d_vertices_ccw(self, vertices):
        """
        Sorts 3D vertices in counter-clockwise order.
         Args:
            vertices (numpy.ndarray): The vertices of a 3D polygon. 

         Returns:
            numpy.ndarray: The sorted vertices.
        """
        # vertices = np.array(vertices)
        center = np.mean(vertices, axis=0)

        # Determine the normal vector of the plane using SVD or cross product.
        # SVD is more robust for approximate planes.
        P = vertices - center
        _, _, V = np.linalg.svd(P)
        # The normal vector is the third column of V (least significant singular value direction)
        normal = V[2, :]

        # Project to a local 2D plane.
        if abs(normal[0]) > abs(normal[1]):
            ref_vec = np.array([0., 1., 0.])
        else:
            ref_vec = np.array([1., 0., 0.])

        u = np.cross(normal, ref_vec)
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)  # v is orthogonal to both normal and u

        # Project the centered points onto the (u, v) plane
        projected_2d = np.array([[np.dot(p, u), np.dot(p, v)] for p in P])

        # Sort in 2D by angle.
        angles = np.arctan2(projected_2d[:, 1], projected_2d[:, 0])
        sorted_indices = np.argsort(angles)

        # Reorder the original 3D vertices.
        sorted_vertices = vertices[sorted_indices]

        return sorted_vertices

    def round_off(self, number, ndigits=8):
        p = 10 ** ndigits
        return (number * p * 2 + 1) // 2 / p


class DummyPointsGenerator:

    def __init__(self, diff=0.5):
        self.diff = diff

    def __iter__(self):
        lower = -self.diff
        upper = 1. + self.diff
        nums = (lower, 0, upper)

        for i in nums:
            for j in nums:
                for k in nums:
                    if i == 0 and j == 0 and k == 0:
                        continue
                    yield (i, j, k)


class ClippingBox(Polygon3DMixin):
    """A class to clip a cube with edges in the range of 0 to 1
    """

    def __init__(self, vor):
        self.vor = vor
        self.clipped_polygons = None

        # Define the six faces of a cube (normal, point on the face)
        self.planes = [
            (np.array([0.0, 0.0, 1.0]), np.array([0., 0., 0])),      # Z min
            (np.array([0.0, 0.0, -1.0]), np.array([0., 0., 1.0])),   # Z max
            (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])),  # X min
            (np.array([-1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])), # X max
            (np.array([0.0, 1.0, 0.0]), np.array([0., 0., 0.])),     # Y min
            (np.array([0.0, -1.0, 0.0]), np.array([0., 1., 0.])),    # Y max
        ]

    def __call__(self, region):
        self.region = region
        return self

    def __iter__(self):
        self.clipped_polygons = []

        # vor.ridge_vertices: Indices of the Voronoi vertices forming each Voronoi ridge.
        for rv in self.vor.ridge_vertices:
            if np.isin(rv, self.region).all():
                # vor.vertices: Coordinates of the Voronoi vertices.
                verts = self.round_off(self.vor.vertices[rv])
                sorted_verts = self.sort_3d_vertices_ccw(verts)

                if clipped_polygon := self.sutherland_hodgman_3d(sorted_verts):
                    clipped_polygon = np.array(clipped_polygon)
                    self.clipped_polygons.append(self.sort_3d_vertices_ccw(clipped_polygon))

                    yield sorted_verts

    def sutherland_hodgman_3d(self, polygon):
        clipped_polygon = polygon

        for normal, point in self.planes:
            if not (clipped_polygon := [p for p in self.clip(clipped_polygon, normal, point)]):
                break

        return clipped_polygon

    def clip(self, polygon, plane_normal, plane_point):
        # new_polygon = []

        for i, p1 in enumerate(polygon):
            p2 = np.array(polygon[(i + 1) % len(polygon)])

            p1_inside = self.is_inside(p1, plane_normal, plane_point)
            p2_inside = self.is_inside(p2, plane_normal, plane_point)

            if p1_inside and p2_inside:
                # Both inner sides: Add only t he endpoint.
                yield self.round_off(p2)
                # new_polygon.append(self.round_off(p2))
            elif p1_inside and not p2_inside:
                # From the inside to outside: Add an intersection.
                yield self.intersection(p1, p2, plane_normal, plane_point)
                # new_polygon.append(self.intersection(p1, p2, plane_normal, plane_point))
            elif not p1_inside and p2_inside:
                # From the outside to inside: Add intersection and p2.
                yield self.intersection(p1, p2, plane_normal, plane_point)
                yield self.round_off(p2)
                # new_polygon.append(self.intersection(p1, p2, plane_normal, plane_point))
                # new_polygon.append(self.round_off(p2))
            else:
                pass

        # return new_polygon

    def is_inside(self, p, plane_normal, plane_point, tolerance=1e-6):
        """Determine whether a point lies on the inside of a plane.
        """
        dot_product = np.dot(p - plane_point, plane_normal)

        if abs(dot_product) < tolerance:
            return True
        if dot_product > 0:
            return True

    def intersection(self, p1, p2, plane_normal, plane_point):
        """Calculate the intersection of a line passing through two points and a plane.
        """
        denom = np.dot(p2 - p1, plane_normal)
        if abs(denom) < 1e-6:
            return p1

        t = np.dot(plane_point - p1, plane_normal) / denom

        return self.round_off(p1 + t * (p2 - p1))


class ConnectVertices:

    pass




class VoronoiGenerator3D:

    def __init__(self, cut_points=15, diff=0.5):
        self.cut_points = cut_points
        self.diff = diff

    def __iter__(self):
        pts = np.random.rand(self.cut_points, 3)
        dummy_pts = np.array([pt for pt in DummyPointsGenerator(self.diff)])
        all_pts = np.concatenate([pts, dummy_pts])
        vor = Voronoi(all_pts)
        clipping_box = ClippingBox()

        # Index of the Voronoi region for each input point
        for region_index in vor.point_region:
            # Indices of the Voronoi vertices forming each Voronoi region.
            region = vor.regions[region_index]

            if -1 not in region and len(region) > 0:
                clipped_regions = [r for r in clipping_box(region)]







if __name__ == '__main__':
    pts = [n for n in DummyPointsGenerator()]
    print(pts)