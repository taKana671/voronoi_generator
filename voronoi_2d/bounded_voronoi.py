import numpy as np
from scipy.spatial import ConvexHull, Voronoi
from shapely.geometry import Polygon


class BoundedVoronoiGenerator:
    """A class that generates vertex coordinates of Voronoi cells clipped to a square

        Args:
            pts (numpy.ndarray): coordinates of points to construct a Voronoi diagram from; default is None.
                If `None` is specified, a number of points equal to `cut_points` will be generated randomly.
            bnd (numpy.ndarray): clipping boundary; default is None.
                If `None` is specified, a square with a side length of 1 is used.
            cut_points (int): number of Voronoi cells; default is 10.
            shrink (float): specify how much to shrink the Voronoi cells.
    """

    def __init__(self, pts=None, bnd=None, cnt_points=10, shrink=0.02):
        self.shrink = shrink
        self.pts = pts
        self.bnd = bnd

        if self.pts is None:
            self.pts = np.random.rand(cnt_points, 2)

        if self.bnd is None:
            self.bnd = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

    def __iter__(self):
        dummy_pts = np.array([[100, 100], [100, -100], [-100, 0]])
        conc_pts = np.concatenate([self.pts, dummy_pts])

        vor = Voronoi(conc_pts)
        bnd_poly = Polygon(self.bnd)

        for i in range(len(conc_pts) - len(dummy_pts)):
            vor_poly = [vor.vertices[v] for v in vor.regions[vor.point_region[i]]]
            poly = Polygon(vor_poly)

            if self.shrink:
                if (shrunk_poly := poly.buffer(-self.shrink)).is_empty:
                    continue
                poly = shrunk_poly

            cell = bnd_poly.intersection(poly)
            yield np.array(cell.exterior.coords[:-1])


class ConvexPolygonGenerator:
    """A class that randomly generates vertex coordinates inside a polygon

        Args:
            bnd (numpy.ndarray): coordinates of the vertices of a polygon
    """

    def __init__(self, bnd):
        self.bnd = bnd

    def get_cnt_points_from_area(self, bnd_hull):
        """Determine the number of Voronoi cells based on the area of the polygon.
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
        """Determine a number based on the area of the polygon, and then randomly generate
           the vertex coordinates of that number within the polygon.
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