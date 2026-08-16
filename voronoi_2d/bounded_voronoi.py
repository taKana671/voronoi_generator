import math

import numpy as np

from scipy.spatial import ConvexHull, Voronoi
from shapely import segmentize
from shapely.geometry import Polygon


class BoundedVoronoiGenerator:
    """A class that performs a Voronoi tessellation of a bounded region.

        Args:
            pts (numpy.ndarray):
                Vertex coordinates of the sites; [[x1, y1], [x2, y2], ...]
                If None is specified, the specified number of cnt_points will be automatically generated.
            bnd (numpy.ndarray): Vertex coordinates of a convex polygon as clipping boundary; default is None.
                If `None` is specified, a square with a side length of 1 is used.
            cut_points (int): Number of sites; default is 10.
            buffer_size_erosion (float):
                Specify how much to shrink the Voronoi cells; must be less than 0.
                If no scaling is required, specify None.
    """

    def __init__(self, pts=None, bnd=None, cnt_points=10, buffer_size_erosion=-0.02):
        self.buffer_size_erosion = buffer_size_erosion
        self.pts = pts
        self.bnd = bnd

        if self.pts is None:
            self.pts = np.random.rand(cnt_points, 2)

        if self.bnd is None:
            self.bnd = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

    def __iter__(self):
        """Generate coordinates of the Voronoi cell vertices.
        """
        dummy_pts = np.array([[100, 100], [100, -100], [-100, 0]])
        conc_pts = np.concatenate([self.pts, dummy_pts])

        vor = Voronoi(conc_pts)
        bnd_poly = Polygon(self.bnd)

        for i in range(len(self.pts)):
            vor_poly = [vor.vertices[v] for v in vor.regions[vor.point_region[i]]]
            poly = Polygon(vor_poly)

            if self.buffer_size_erosion:
                if (shrunk_poly := poly.buffer(self.buffer_size_erosion)).is_empty:
                    continue
                poly = shrunk_poly

            if (cell := bnd_poly.intersection(poly)).is_empty:
                continue

            yield np.array(cell.exterior.coords[:-1])


class VoronoiSitesGenerator:
    """A class that randomly generates points inside a convex polygon

        Args:
            bnd (numpy.ndarray): Coordinates of convex polygon vertices
    """

    def __init__(self, bnd):
        self.bnd = bnd

    def get_cnt_points_from_area(self, bnd_hull):
        """Determine the number of Voronoi sites based on the area of the convex polygon.
            Args:
                bnd_hull (scipy.spatial.ConvexHull)
        """
        if (area := bnd_hull.volume) < 0.02:
            return 2
        elif area < 0.03:
            return 3
        elif area < 0.04:
            return 4
        elif area < 0.05:
            return 5
        elif area < 0.06:
            return 6
        elif area < 0.07:
            return 7
        elif area < 0.08:
            return 8
        elif area < 0.09:
            return 9
        else:
            return 10

    def __iter__(self):
        """Determine the number of sites in a Voronoi diagram based on the area of
           the convex polygon(self.bnd), and generate the coordinates of those sites.
        """
        bnd_hull = ConvexHull(self.bnd)
        cnt_pts = self.get_cnt_points_from_area(bnd_hull)

        # A matrix representing the boundaries of a domain
        bnd_tmp = bnd_hull.equations
        bnd_mat = np.matrix(bnd_tmp)
        a_bnd = np.array(bnd_mat[:, 0:2])
        b_bnd = np.array(bnd_mat[:, 2])

        # The rectangle surrounding the area
        xmin = np.min(self.bnd[:, 0])
        xmax = np.max(self.bnd[:, 0])
        ymin = np.min(self.bnd[:, 1])
        ymax = np.max(self.bnd[:, 1])

        i = 0

        while i < cnt_pts:
            pt = np.random.rand(2)
            pt[0] = xmin + (xmax - xmin) * pt[0]
            pt[1] = ymin + (ymax - ymin) * pt[1]
            n = len(self.bnd)

            # Determine whether a point lies inside the polygon
            if (np.round(np.dot(a_bnd, pt.transpose()), n) <= np.round(-b_bnd.transpose(), n)).all():
                yield pt
                i += 1


class RoundedVoronoiGenerator:
    """A class that partitions a closed region into Voronoi cells with rounded corners.

        Args:
            pts (numpy.ndarray):
                Vertex coordinates of the sites; [[x1, y1], [x2, y2], ...]
                If None is specified, the coordinates of number specified by cnt_points will be automatically generated.
            bnd (numpy.ndarray): Vertex coordinates of a convex polygon as clipping boundary; default is None.
                If None is specified, a square with a side length of 1 is used.
            cut_points (int): Number of sites; default is 10.
            buffer_size_erosion (float):
                Specify the distance to scale inward within the polygon; must be less than 0.
                Default is -0.04.
            buffer_size_dilation (float):
                Specify the distance to expand beyond the polygon; must be greater than 0.
                Default is 0.03.
            quad_segs (int):
                Specify how many line segments to use to divide a single quadrant (90 degrees).
                Default is 16.
            segment_length (float):
                Specify the length at which to divide the straight sections (excluding rounded corners).
                If None is specified, the length is automatically calculated based on quad_segs.
    """

    def __init__(self, pts=None, bnd=None, cnt_points=10, buffer_size_erosion=-0.04,
                 buffer_size_dilation=0.03, quad_segs=16, segment_length=None):
        self.pts = pts
        self.bnd = bnd
        # self.cnt_points = cnt_points
        self.buffer_size_erosion = buffer_size_erosion
        self.buffer_size_dilation = buffer_size_dilation
        self.quad_segs = quad_segs
        self.segment_length = segment_length or self.calc_segment_length()

        if self.pts is None:
            self.pts = np.random.rand(cnt_points, 2)

        if self.bnd is None:
            self.bnd = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

    def calc_segment_length(self):
        """Based on quad_segs, automatically calculate the vertex spacing
           (max_segment_length) for straight segments.
        """
        arc_length_90 = (2 * math.pi * self.buffer_size_dilation) / 4
        single_seg_length = arc_length_90 / self.quad_segs
        segment_length = single_seg_length * 2.5
        return segment_length

    def __iter__(self):
        """Generate coordinates of the rounded Voronoi cell vertices.
        """
        dummy_pts = np.array([[100, 100], [100, -100], [-100, 0]])
        conc_pts = np.concatenate([self.pts, dummy_pts])

        vor = Voronoi(conc_pts)
        bnd_poly = Polygon(self.bnd)

        for i in range(len(self.pts)):
            vor_poly = [vor.vertices[v] for v in vor.regions[vor.point_region[i]]]
            poly = Polygon(vor_poly)

            # Make the polygon slightly smaller than the original size.
            if (shrunk_poly := poly.buffer(self.buffer_size_erosion)).is_empty:
                continue

            poly = shrunk_poly
            cell = bnd_poly.intersection(poly)

            # Round the corners and restore it slightly to its original size.
            cell = cell.buffer(self.buffer_size_dilation, join_style='round', quad_segs=self.quad_segs)
            # Increase the mumber of vertices on a line.
            cell = segmentize(cell, max_segment_length=self.segment_length)

            yield np.array(cell.exterior.coords[:-1])
