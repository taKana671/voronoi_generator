import numpy as np
from scipy.spatial import Voronoi

from ..polygon3d_mixin import Polygon3DMixin
from pprint import pprint

NDIGITS = 8

# np.seterr(all='raise')


class Sphere:
    """A class representing a clipping spherical region.

        Args:
            center (numpy.ndarray): the center of a sphere.
            radius (float): the sphere radius.
    """

    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def dummy_points(self, num_points=30, buffer_rate=2.):
        # Make the radius larger than that of the center sphere.
        dummy_radius = self.radius * buffer_rate

        # Generate points randomly (normalized using a normal distribution).
        points = np.random.randn(num_points, 3)
        points /= np.linalg.norm(points, axis=1)[:, np.newaxis]

        dummy_points = points * dummy_radius + np.array(self.center)
        return dummy_points

    def is_inside(self, pt):
        """Returns True if the coordinates of a point lie inside the sphere, and False otherwise.
            pt (numpy.ndarray): the coordinates of a point
        """
        return sum((pt[i] - self.center[i]) ** 2 for i in range(3)) ** 0.5 <= self.radius ** 2

    def intersect(self, p1, p2):
        d = p2 - p1
        f = p1 - self.center

        a = np.dot(d, d)
        b = 2 * np.dot(f, d)
        c = np.dot(f, f) - self.radius ** 2

        discriminant = b ** 2 - 4 * a * c

        if discriminant < 0:
            return None

        sqrt_disc = discriminant ** 0.5
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)

        for t in [t1, t2]:
            if 0 <= t <= 1:
                return p1 + t * d

        return None


class Clipping(Polygon3DMixin):
    """A class that uses the Sutherland-Hodgman algorithm to generate the vertex coordinates
       of a voronoi cell (polyhedron) clipped to a sphere.

        Arges:
            vor (scipy.spatial._qhull.voronoi): the voronoi diagram.
            sphere (Sphere): clipping region.
    """

    def __init__(self, vor, sphere):
        self.vor = vor
        self.sphere = sphere

    def __call__(self, region):
        self.region = region
        return self

    def __iter__(self):
        """Generate the vertices of a polygon clipped to a sphere
           and the intersection points between the polygon and the sphere.
        """
        for rv in self.vor.ridge_vertices:
            # rv means indices of the voronoi vertices forming each voronoi ridge.
            if np.isin(rv, self.region).all():
                sorted_verts = self.sort_3d_vertices_ccw(self.vor.vertices[rv])
                clipped_verts, do_intersects = [], []

                for pt, pt_or_ip in self.clip(sorted_verts):
                    clipped_verts.append(pt)
                    do_intersects.append(pt_or_ip)

                if clipped_verts:
                    clipped_verts = self.round_off(np.array(clipped_verts), NDIGITS)

                    mask = np.array(do_intersects)
                    intersections = clipped_verts[mask]

                    clipped_verts = self.sort_3d_vertices_ccw(clipped_verts)

                    yield (clipped_verts, intersections)

    def clip(self, vertices):
        """Clipping by using the 3D version of the Sutherland-Hodgman algorithm.
           If a line segment consisting of two vertices of a polygon (in counterclockwise order)
           intersects a sphere, return the intersection point and True.
           If there is no intersection, return the starting point and False.

            Args:
                vertices (numpy.ndarray): vertices of a 3D polygon; sorted in counterclockwise order.
        """
        length = len(vertices)

        for i, p1 in enumerate(vertices):
            p2 = vertices[(i + 1) % length]

            p1_inside = self.sphere.is_inside(p1)
            p2_inside = self.sphere.is_inside(p2)

            if p1_inside and p2_inside:
                # Both inner sides: Add only the endpoint.
                yield (p2, False)

            elif p1_inside and not p2_inside:
                # From the inside to outside: Add an intersection.
                ip = self.sphere.intersect(p1, p2)
                yield (ip, True)

            elif not p1_inside and p2_inside:
                # From the outside to inside: Add intersection and p2.
                ip = self.sphere.intersect(p1, p2)
                yield (ip, True)
                yield (p2, False)
            else:
                # Both outside: Do nothing.
                pass


class VoronoiClipped2Sphere(Polygon3DMixin):
    """A class that generates vertex coordinates for each voronoi cell clipped to a cube.

        Args:
            cut_points(int): the number of polyhedrons to divide a cube into.
            cube_size (float): length of a cube's edge.
            diff (float): how far from the vertices of the cube the dummy points should be placed.
    """

    def __init__(self, center, radius=1., cut_points=30):
        self.cut_points = cut_points
        self.sphere = Sphere(center, radius)

    def __iter__(self):
        """Generate the following two arrays.
            polygons_clipped (numpy.ndarray): the vertices on each face of a polyhedron.
            polygon_vertices (numpy.ndarray):
                vertices to be converted to the spherical face;
                if the conversion is not necessary, generate an empty list.
        """

        rng = np.random.default_rng()
        pts = rng.uniform(0, self.sphere.radius * 2, (self.cut_points, 3))
        # pts = rng.uniform(-self.sphere.radius, self.sphere.radius, (self.cut_points, 3))

        dummy_pts = self.sphere.dummy_points()
        all_pts = np.concatenate([pts, dummy_pts])

        vor = Voronoi(all_pts)
        clipping = Clipping(vor, self.sphere)

        # Index of the voronoi region for each input point
        for region_index in vor.point_region:
            # Indices of the voronoi vertices forming each voronoi region.
            region = vor.regions[region_index]

            if -1 not in region and len(region) > 0:
                polygons_clipped, polygon_vertices = [], []

                for clipped, intersections in clipping(region):
                    polygons_clipped.append(clipped)

                    if intersections.size > 0:
                        polygon_vertices.append(intersections)

                if polygon_vertices:
                    polygon_vertices = np.unique(np.vstack(polygon_vertices), axis=0)
                    polygon_vertices = self.sort_3d_vertices_ccw(np.array(polygon_vertices))
                    polygons_clipped.append(polygon_vertices)

                if polygons_clipped:
                    yield polygons_clipped, polygon_vertices
