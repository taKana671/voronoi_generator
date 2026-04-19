import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import time


class Polygon3DMixin:

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

    def round_off(self, number, ndigits=8):
        p = 10 ** ndigits
        return (number * p * 2 + 1) // 2 / p

    def unique_close(self, vertices, atol=1e-12):
        for i in range(len(vertices) - 1):
            v1 = vertices[i]
            v2 = vertices[i + 1]

            if np.all(np.isclose(v1, v2, atol=atol)):
                continue

            yield v1

        yield vertices[-1]

    def cross(self, a, b):
        """Using numpy.cross on a small numpy.ndarray with shape=(3, ) caused performance issues,
           so use the Python implementation of cross instead.
        """
        return np.array([
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]
        ])


class Cube:

    def __init__(self, size=1.):
        self.size = size
        self.vertices = self.define_vertices()
        self.planes = self.define_planes()
        self.edges = self.define_edges()

    def define_vertices(self):
        vertices = np.array([
            [0, 0, 0],
            [0, 0, self.size],
            [self.size, 0, 0],
            [self.size, 0, self.size],
            [self.size, self.size, 0],
            [self.size, self.size, self.size],
            [0, self.size, 0],
            [0, self.size, self.size]
        ])

        return vertices

    def define_planes(self):
        """Define the six faces of a cube (normal, point on the face)
        """
        planes = [
            (np.array([0.0, 0.0, 1.0]), self.vertices[0]),   # Z min
            (np.array([0.0, 0.0, -1.0]), self.vertices[1]),  # Z max
            (np.array([1.0, 0.0, 0.0]), self.vertices[0]),   # X min
            (np.array([-1.0, 0.0, 0.0]), self.vertices[2]),  # X max
            (np.array([0.0, 1.0, 0.0]), self.vertices[0]),   # Y min
            (np.array([0.0, -1.0, 0.0]), self.vertices[6]),  # Y max
        ]

        return planes

    def define_edges(self):
        indices = [
            [5, 7], [7, 1], [1, 3], [3, 5],
            [4, 6], [6, 0], [0, 2], [2, 4],
            [3, 1], [1, 0], [0, 2], [2, 3],
            [5, 3], [3, 2], [2, 4], [4, 5],
            [5, 7], [7, 6], [6, 4], [4, 5],
            [1, 7], [7, 6], [6, 0], [0, 1]
        ]

        edges = [self.vertices[idx] for idx in indices]
        return edges

    def dummy_points(self, diff=0.5):
        lower = -diff
        upper = self.size + diff
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
        self.cube = cube

    def __call__(self, region):
        self.region = region
        return self

    def __iter__(self):
        # vor.ridge_vertices: Indices of the Voronoi vertices forming each Voronoi ridge.
        for rv in self.vor.ridge_vertices:
            if np.isin(rv, self.region).all():
                # vor.vertices: Coordinates of the Voronoi vertices.
                verts = self.round_off(self.vor.vertices[rv])
                sorted_verts = self.sort_3d_vertices_ccw(verts)

                if clipped_polygon := self.sutherland_hodgman_3d(sorted_verts):
                    clipped_polygon = self.sort_3d_vertices_ccw(np.array(clipped_polygon))
                    yield (sorted_verts, clipped_polygon)

    def sutherland_hodgman_3d(self, polygon):
        clipped_polygon = polygon

        for normal, point in self.cube.planes:
            if not (clipped_polygon := [p for p in self.clip(clipped_polygon, normal, point)]):
                break

        return clipped_polygon

    def clip(self, polygon, plane_normal, plane_point):
        for i, p1 in enumerate(polygon):
            p2 = np.array(polygon[(i + 1) % len(polygon)])

            p1_inside = self.is_inside(p1, plane_normal, plane_point)
            p2_inside = self.is_inside(p2, plane_normal, plane_point)

            if p1_inside and p2_inside:
                # Both inner sides: Add only t he endpoint.
                yield self.round_off(p2)
            elif p1_inside and not p2_inside:
                # From the inside to outside: Add an intersection.
                yield self.intersection(p1, p2, plane_normal, plane_point)
            elif not p1_inside and p2_inside:
                # From the outside to inside: Add intersection and p2.
                yield self.intersection(p1, p2, plane_normal, plane_point)
                yield self.round_off(p2)
            else:
                pass

    def is_inside(self, p, plane_normal, plane_point, tolerance=1e-6):
        """Determine whether a point lies on the inside of a plane.
        """
        dot_product = np.dot(p - plane_point, plane_normal)

        if abs(dot_product) < tolerance:
            return True
        if dot_product > 0:
            return True

    def intersection(self, p1, p2, plane_normal, plane_point, tolerance=1e-6):
        """Calculate the intersection of a line passing through two points and a plane.
        """
        denom = np.dot(p2 - p1, plane_normal)
        if abs(denom) < tolerance:
            return p1

        t = np.dot(plane_point - p1, plane_normal) / denom

        return self.round_off(p1 + t * (p2 - p1))


class Intersect(Polygon3DMixin):

    def __init__(self, polygons, cube):
        self.polygons = polygons
        self.cube = cube

    def __iter__(self):
        for polygon in self.polygons:
            center = np.mean(polygon, axis=0)

            for i, p1 in enumerate(polygon):
                p2 = polygon[(i + 1) % len(polygon)]

                for start, end in self.cube.edges:
                    if (intersection := self.get_intersection(
                            p1, p2, center, start, end, polygon)) is not None:
                        yield intersection

    def get_intersection(self, p1, p2, center, l1, l2, polygon):
        if (intersection := self.intersect_line_plane(p1, p2, center, l1, l2)) is not None:
            if self.is_inside_plane(intersection, polygon):
                return intersection

        return None

    def intersect_line_plane(self, v0, v1, v2, p1, p2):
        e1 = v1 - v0
        e2 = v2 - v0

        d = p2 - p1
        # Plane normal
        n = self.cross(e1, e2)

        # Checking for intersection between a line segment and a plane.
        det = np.dot(d, n)
        if abs(det) < 1e-6:
            return None

        # Distance to the intersection on the plane
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

        if abs(area_target - area_polygon) < tolerance:
            return True

    def calc_triangle_area(self, p0, p1, p2):
        v1 = p1 - p0
        v2 = p2 - p0

        c = self.cross(v1, v2)
        area = (c[0] ** 2 + c[1] ** 2 + c[2] ** 2) ** 0.5

        return area

    def calc_polygon_area(self, polygon):
        p0 = polygon[0]
        area = sum(self.calc_triangle_area(p0, polygon[i], polygon[i + 1])
                   for i in range(1, len(polygon) - 1))

        return area


class Corners(Polygon3DMixin):

    def __init__(self, cube, org_polygons, clipped_polygons, intersections):
        self.cube = cube
        self.org_polygons = org_polygons
        self.clipped_polygons = clipped_polygons
        self.intersections = intersections

    def __iter__(self):
        hull = ConvexHull(np.concatenate(self.org_polygons))
        ipts = np.array(self.intersections)

        for corner in self.cube.vertices:
            # If the polyhedron contains a vertex before clipping, include that vertex.
            if np.all(np.dot(hull.equations[:, :-1], corner) + hull.equations[:, -1] <= 1e-5):
                yield corner
                continue

            # If vertices lies on three edges that form a corner, and the triangular face formed by
            # those vertices is contained within the clipped polyhedron, include the vertex of the corner.
            if len(x := ipts[(ipts[:, 1] == corner[1]) & (ipts[:, 2] == corner[2])]) > 0 and \
                    len(y := ipts[(ipts[:, 0] == corner[0]) & (ipts[:, 2] == corner[2])]) > 0 and \
                    len(z := ipts[(ipts[:, 0] == corner[0]) & (ipts[:, 1] == corner[1])]) > 0:

                funcs = [np.min if corner[i] == 0 else np.max for i in range(3)]
                tri = np.array([func(p, axis=0) for func, p in zip(funcs, [x, y, z])])

                if not self.do_contain_face(tri):
                    yield corner

    def do_contain_face(self, tri):
        """Check whether the clipped polyhedron contains the target triangular face
           by comparing vertex coordinates.
            Args:
                tri (numpy.ndarray): The vertices of a triangle.
        """
        tri = self.sort_3d_vertices_ccw(tri)

        return any(np.allclose(polygon, tri, rtol=1e-05, atol=1e-15)
                   for polygon in self.clipped_polygons if len(polygon) == 3)


class Faces(Polygon3DMixin):

    def __init__(self, clipped_polygons, intersections, corners):
        self.clipped_polygons = clipped_polygons
        self.intersections = intersections
        self.corners = corners

    def __iter__(self):
        new_vertices = self.intersections + self.corners

        for i in range(3):
            for val in [0.0, 1.0]:
                new_polygon = []

                for polygon in self.clipped_polygons:

                    for pv in polygon:
                        for pi in self.intersections:
                            if np.all(np.isclose(pv, pi, atol=1e-12)):
                                pv[:] = pi[:]
                                break

                    if (vertices := polygon[polygon[:, i] == val]).size > 0:
                        new_polygon.extend(vertices)

                    if included := [v for v in new_vertices if v[i] == val]:
                        new_polygon.extend(included)

                if new_polygon:
                    uq_vertices = np.unique(new_polygon, axis=0)
                    sorted_vertices = self.sort_3d_vertices_ccw(uq_vertices)
                    yield sorted_vertices


class VoronoiGenerator3D(Polygon3DMixin):

    def __init__(self, cut_points=10, diff=0.5):
        self.cut_points = cut_points
        self.diff = diff

    def __iter__(self):
        cube = Cube(size=1.)
        pts = np.random.rand(self.cut_points, 3)
        # pts = np.array([
        #     [0.84497989, 0.3136216, 0.63398512],
        #     [0.19015443, 0.2373259, 0.64581667]
        # ])

        dummy_pts = np.array([pt for pt in cube.dummy_points()])
        all_pts = np.concatenate([pts, dummy_pts])
        vor = Voronoi(all_pts)
        clipping = Clipping(vor, cube)

        # Index of the Voronoi region for each input point
        for region_index in vor.point_region:
            # Indices of the Voronoi vertices forming each Voronoi region.
            region = vor.regions[region_index]

            if -1 not in region and len(region) > 0:
                polygons_clipped, polygons_original, corners = [], [], []

                for org, clipped in clipping(region):
                    polygons_clipped.append(clipped)
                    polygons_original.append(org)

                if len(intersections := [ip for ip in Intersect(polygons_original, cube)]) > 0:
                    intersections = [ip for ip in self.unique_close(np.unique(intersections, axis=0))]
                    corners = [c for c in Corners(cube, polygons_original, polygons_clipped, intersections)]

                if new_polygons := [poly for poly in Faces(polygons_clipped, intersections, corners)]:
                    polygons_clipped.extend(new_polygons)

                yield polygons_clipped


def main():
    start = time.perf_counter()
    polyhedrons = [polyhedron for polyhedron in VoronoiGenerator3D(cut_points=30)]
    print(f'Took {time.perf_counter() - start}')  # Took 2.6943330999984028

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # 1行目1列の1番目という意味で、subplot(1,1,1)でも同じ

    for poly in polyhedrons:
    # for poly in new_polyhedrons:
        polygon = Poly3DCollection(
            poly,
            alpha=0.6,
            facecolors=np.random.uniform(0, 1, 3),
            linewidth=0.8,
            edgecolors='gray'
        )
        ax.add_collection3d(polygon)

    ax.set_xlim([0, 1.])
    ax.set_ylim([0, 1.])
    ax.set_zlim([0, 1.])
    plt.show()


if __name__ == '__main__':
    main()
    # cube = Cube()
    # cube.define_edges()
    # pts = [n for n in cube.dummy_points()]
    # print(pts)