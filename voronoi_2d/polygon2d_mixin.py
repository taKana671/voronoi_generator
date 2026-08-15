import shapely
import numpy as np
from shapely.constructive import maximum_inscribed_circle
from shapely.geometry import Polygon

from ..polygon_mixin import PolygonMixin


class Polygon2DMixin(PolygonMixin):

    def get_max_inscribed_circle(self, pts):
        """Get the radius and center of the inscribed circle.
            Args:
                pts (Numpy.ndarray): Iregular polygon vertices
        """
        poly = Polygon(pts)
        mic = maximum_inscribed_circle(poly)
        center = mic.coords[0]
        radius = mic.length

        return center, radius

    def sort_counter_clockwise(self, arr):
        center = np.mean(arr, axis=0)
        angles = np.arctan2(arr[:, 1] - center[1], arr[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        sorted_pts = arr[sorted_indices]
        return sorted_pts

    def round_polygon_corners(self, vertices, buffer_size_erosion=None, buffer_size_dilation=0.03,
                              quad_segs=16, segment_length=None):
        """Rounds the corners of a convex polygon to return Numpy.ndarray of vertex coordinates.
        """

        poly = Polygon(vertices)

        if buffer_size_erosion is not None:
            if (shrunk_poly := poly.buffer(buffer_size_erosion)).is_empty:
                return None

            poly = shrunk_poly

        cell = poly.buffer(buffer_size_dilation, join_style='round', quad_segs=quad_segs)

        if segment_length is not None:
            cell = shapely.segmentize(cell, max_segment_length=segment_length)

        arr = np.array(cell.exterior.coords[:-1])
        return arr




    # def round_corners(self, pts, buffer_size_dilation, quad_seg):
    #     """Rounds the corners of a convex polygon to return Numpy.ndarray of vertex coordinates.
    #          Args:
    #             pts (Numpy.ndarray): polygon vertex coordinate
    #     """
    #     arc_length_90 = (2 * math.pi * buffer_size_dilation) / 4
    #     single_seg_length = arc_length_90 / quad_seg
    #     segment_length = single_seg_length * 2.5


    #     poly = Polygon(pts)
    #     cell = poly.buffer(buffer_size_dilation, join_style='round', quad_segs=quad_seg)

    #     cell = shapely.segmentize(cell, max_segment_length=segment_length)
        

    #     arr = np.array(cell.exterior.coords[:-1])
    #     return arr
