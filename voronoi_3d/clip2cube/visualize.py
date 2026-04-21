import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .clip2cube import VoronoiClipped2Cube


def visualize(cut_points=10, cube_size=1., diff=.5, alpha=.6):
    polyhedrons = [polyhedron for polyhedron in VoronoiClipped2Cube(cut_points, cube_size, diff)]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for poly in polyhedrons:
        polygon = Poly3DCollection(
            poly,
            alpha=alpha,
            facecolors=np.random.uniform(0, 1, 3),
            linewidth=0.8,
            edgecolors='gray'
        )
        ax.add_collection3d(polygon)

    ax.set_xlim([0, cube_size])
    ax.set_ylim([0, cube_size])
    ax.set_zlim([0, cube_size])
    plt.show()