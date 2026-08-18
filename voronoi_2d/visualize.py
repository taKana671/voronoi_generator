import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from .bounded_voronoi import BoundedVoronoiGenerator, VoronoiSitesGenerator
from .bounded_voronoi import RoundedVoronoiGenerator


def visualize(pts=None, bnd=None, cnt_points=20, buffer_size_erosion=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for region in BoundedVoronoiGenerator(
            pts, bnd, cnt_points, buffer_size_erosion=buffer_size_erosion):
        ax.add_patch(Polygon(
            region,
            facecolor=np.random.uniform(0, 1, 3),
            alpha=0.6,
            edgecolor='gray')
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.show()


def convex_voronoi(vertices, cnt_points=10):
    pts = [pt for pt in VoronoiSitesGenerator(vertices)]
    visualize(pts=pts, bnd=vertices, cnt_points=cnt_points)


def visualize_rounded_voronoi(pts=None, bnd=None, cnt_points=20,
                              buffer_size_erosion=-0.04, buffer_size_dilation=0.03):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for region in RoundedVoronoiGenerator(pts, bnd,
                                          cnt_points=cnt_points,
                                          buffer_size_erosion=buffer_size_erosion,
                                          buffer_size_dilation=buffer_size_dilation):
        ax.add_patch(Polygon(
            region,
            facecolor=np.random.uniform(0, 1, 3),
            alpha=0.6,
            edgecolor='gray')
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.show()


def convex_rounded_voronoi(vertices, cnt_points=10,
                           buffer_size_erosion=-0.04, buffer_size_dilation=0.03):
    pts = [pt for pt in VoronoiSitesGenerator(vertices)]
    visualize_rounded_voronoi(
        pts=pts,
        bnd=vertices,
        cnt_points=cnt_points,
        buffer_size_erosion=buffer_size_erosion,
        buffer_size_dilation=buffer_size_dilation
    )
