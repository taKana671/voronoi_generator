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


class Cube:

    def __init__(self, edge=1.):
        self.edge = edge
        # self.planes = self.create_planes()

    def define_planes(self):
        # Define the six faces of a cube (normal, point on the face)
        planes = [
            (np.array([0.0, 0.0, 1.0]), np.array([0., 0., 0.])),     # Z min
            (np.array([0.0, 0.0, -1.0]), np.array([0., 0., self.edge])),  # Z max
            (np.array([1.0, 0.0, 0.0]), np.array([0., 0., 0.])),     # X min
            (np.array([-1.0, 0.0, 0.0]), np.array([self.edge, 0., 0.])),  # X max
            (np.array([0.0, 1.0, 0.0]), np.array([0., 0., 0.])),     # Y min
            (np.array([0.0, -1.0, 0.0]), np.array([0., self.edge, 0.])),  # Y max
        ]

        return planes

    def define_edges(self):
        vertices = np.array([
            [0, 0, 0],
            [0, 0, self.edge],
            [self.edge, 0, 0],
            [self.edge, 0, self.edge],
            [self.edge, self.edge, 0],
            [self.edge, self.edge, self.edge],
            [0, self.edge, 0],
            [0, self.edge, self.edge]
        ])

        indices = [
            [5, 7], [7, 1], [1, 3], [3, 5],
            [4, 6], [6, 0], [0, 2], [2, 4],
            [3, 1], [1, 0], [0, 2], [2, 3],
            [5, 3], [3, 2], [2, 4], [4, 5],
            [5, 7], [7, 6], [6, 4], [4, 5],
            [1, 7], [7, 6], [6, 0], [0, 1]
        ]

        edges = [vertices[idx] for idx in indices]
        return edges

    def dummy_points(self, diff=0.5):
        lower = -diff
        upper = self.edge + diff
        nums = (lower, 0, upper)

        for i in nums:
            for j in nums:
                for k in nums:
                    if i == 0 and j == 0 and k == 0:
                        continue
                    yield (i, j, k)


class Clipping(Polygon3DMixin):
    """A class to clip a cube with edges in the range of 0 to 1
    """

    def __init__(self, vor, cube):
        self.vor = vor
        # self.clipped_polygons = None
        self.cube_planes = cube.define_planes()

    def __call__(self, region):
        self.region = region
        return self

    def __iter__(self):
        # self.clipped_polygons = []

        # vor.ridge_vertices: Indices of the Voronoi vertices forming each Voronoi ridge.
        for rv in self.vor.ridge_vertices:
            if np.isin(rv, self.region).all():
                # vor.vertices: Coordinates of the Voronoi vertices.
                verts = self.round_off(self.vor.vertices[rv])
                sorted_verts = self.sort_3d_vertices_ccw(verts)

                if clipped_polygon := self.sutherland_hodgman_3d(sorted_verts):
                    clipped_polygon = np.array(clipped_polygon)

                    # self.clipped_polygons.append(self.sort_3d_vertices_ccw(clipped_polygon))

                    yield (sorted_verts, clipped_polygon)

    def sutherland_hodgman_3d(self, polygon):
        clipped_polygon = polygon

        for normal, point in self.cube_planes:
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


class Intersect(Polygon3DMixin):

    def __init__(self, cube):
        self.cube_edges = cube.define_edges()

    def __call__(self, polygons):
        self.polygons = polygons
        return self

    def __iter__(self):
        # results = []

        for polygon in self.polygons:
            center = np.sum(polygon, axis=0) / len(polygon)

            for i in range(len(polygon)):
                p1 = np.array(polygon[i])
                p2 = np.array(polygon[(i + 1) % len(polygon)])

                for start, end in self.cube_edges:
                    if (intersection := self.get_intersection(
                            p1, p2, center, start, end, polygon)) is not None:
                        yield intersection

    def get_intersection(self, p1, p2, center, l1, l2, polygon):
        if (intersection := self.intersect_line_plane(p1, p2, center, l1, l2)) is not None:
            if self.is_inside_plane(intersection, polygon):
                return self.round_off(intersection)

        return None

    def intersect_line_plane(self, v0, v1, v2, p1, p2):
        e1 = v1 - v0
        e2 = v2 - v0

        d = p2 - p1
        # 平面法線
        n = np.cross(e1, e2)

        # 線分と平面の交差判定
        det = np.dot(d, n)
        if abs(det) < 1e-6:
            return None

        # 平面上の交点までの距離
        t = np.dot(v0 - p1, n) / det
        if t < 0.0 or t > 1.0:
            return None

        intersection = p1 + t * d
        return self.round_off(intersection)

    def is_inside_plane(self, intersection, polygon, tolerance=1e-6):
        size = len(polygon)
        area_polygon = self.calc_polygon_area(polygon)
        area_target = sum(self.calc_triangle_area(
            intersection, polygon[i], polygon[(i + 1) % size]) for i in range(size))
        # area = 0

        # for i in range(len(polygon)):
        #     p1 = np.array(polygon[i])
        #     p2 = np.array(polygon[(i + 1) % len(polygon)])
        #     area += calc_triangle_area(intersection, p1, p2)

        # if abs(area - area_polygon) < tolerance:
        if abs(area_target - area_polygon) < tolerance:
            return True

    def calc_triangle_area(self, p0, p1, p2):
        v1 = p1 - p0
        v2 = p2 - p0
        c = np.cross(v1, v2)
        area = sum(c ** 2) ** 0.5
        return area

    def calc_polygon_area(self, polygon):
        p0 = polygon[0]
        area = sum(self.calc_triangle_area(p0, polygon[i], polygon[i + 1])
                   for i in range(1, len(polygon) - 1))
        # area = 0

        # for i in range(1, len(polygon) - 1):
        #     area += self.calc_triangle_area(p0, polygon[i], polygon[i + 1])

        return area




class VoronoiGenerator3D:

    def __init__(self, cut_points=15, diff=0.5):
        self.cut_points = cut_points
        self.diff = diff

    def __iter__(self):
        cube = Cube(edge=1.)
        intersect = Intersect(cube) 

        pts = np.random.rand(self.cut_points, 3)
        dummy_pts = np.array([pt for pt in cube.dummy_points()])
        all_pts = np.concatenate([pts, dummy_pts])
        vor = Voronoi(all_pts)
        clipping = Clipping(vor, cube)

        # Index of the Voronoi region for each input point
        for region_index in vor.point_region:
            # Indices of the Voronoi vertices forming each Voronoi region.
            region = vor.regions[region_index]

            if -1 not in region and len(region) > 0:
                # org_polygon, clipped_polygon = zip(*((org, clipped) for org, clipped in clipping(region)))
                org_polygons = []
                clipped_polygons = []

                for org, clipped in clipping(region):
                    org_polygons.append(org)
                    clipped_polygons.append(clipped)

                if len(intersections := [ip for ip in intersect(org_polygons)]) > 0:
                    pass

               




if __name__ == '__main__':
    cube = Cube()
    cube.define_edges()
    # pts = [n for n in cube.dummy_points()]
    # print(pts)