import random

import cv2
import numpy as np
import shapely
from shapely import Polygon
from shapely.constructive import maximum_inscribed_circle

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

    def find_convex_polygon_in_irregular_region(self, img, within=0.8, vertex_cnt=5):
        """Find the vertex coordinates of a convex polygon within an irregular region.
            Args:
                img (Numpy.ndarray): A black-and-white binary 8bit image
                within (float):
                    Specify the percentage of the radius within which to create a convex polygon.
                    Default is 0.8.
                vertex_cnt (int): The number of a convex polygon; default is 5.
        """
        if img.ndim == 3:
            img = img[:, :, 0]
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the radius and center of the greatest inscribed circle.
        dist_transform = cv2.distanceTransform(img, cv2.DIST_L2, 5)
        _, max_val, _, max_loc = cv2.minMaxLoc(dist_transform)
        cx, cy = max_loc
        radius = max_val * within

        angles = sorted([random.uniform(0, 2 * np.pi) for _ in range(vertex_cnt)])
        vertices = []

        for angle in angles:
            x = int(cx + radius * np.cos(angle))
            y = int(cy + radius * np.sin(angle))
            vertices.append([x, y])

        region = np.array(vertices, dtype=np.float64)
        return (cx, cy), radius, region

    def get_max_distance_from_center(self, verts):
        """Calculate the center point from the vertex coordinates that form a convex polygon,
            and retrieve the coordinates of the vertex farthest from the center point
            Args:
                verts (Numpy.ndarray): vertex coordinates that form a convex polygon
        """
        center = np.mean(verts, axis=0)
        distances = np.sum((verts - center) ** 2, axis=1)
        max_distance = np.max(distances) ** 0.5
        return center, max_distance