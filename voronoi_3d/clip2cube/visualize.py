import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .clip2cube import VoronoiClip2Cube


def visualize(cut_points=30, cube_size=1., diff=.5, alpha=.6):
    polyhedrons = [polyhedron for polyhedron in VoronoiClip2Cube(cut_points, cube_size, diff)]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal', adjustable='box')
    offset = np.full(3, cube_size / 2)
    rng = np.random.default_rng()

    for polyhedron in polyhedrons:
        polygon = Poly3DCollection(
            [poly + offset for poly in polyhedron],
            alpha=alpha,
            facecolors=rng.uniform(0, 1, 3),
            linewidth=0.8,
            edgecolors='gray'
        )
        ax.add_collection3d(polygon)

    ax_range = [0, cube_size]
    ax.set_xlim(ax_range)
    ax.set_ylim(ax_range)
    ax.set_zlim(ax_range)
    plt.show()