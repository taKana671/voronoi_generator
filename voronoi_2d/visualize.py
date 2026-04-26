import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from .bounded_voronoi import BoundedVoronoiGenerator, ConvexPolygonGenerator


def visualize(pts=None, bnd=None, cnt_points=10):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for region in BoundedVoronoiGenerator(pts, bnd, cnt_points, shrink=0.):
        ax.add_patch(Polygon(
            region,
            facecolor=np.random.uniform(0, 1, 3),
            alpha=0.8,
            edgecolor='gray')
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.show()


def visualize_bounded_voronoi(cnt_points=20):
    visualize(cnt_points=cnt_points)


def visualize_convex_voronoi(vertices):
    pts = [pt for pt in ConvexPolygonGenerator(vertices)]
    visualize(pts=pts, bnd=vertices)