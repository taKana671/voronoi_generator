import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .clip2sphere import VoronoiClip2Sphere


def visualize(cut_points=30, alpha=.6):
    polyhedrons = [polyhedron for polyhedron, _ in VoronoiClip2Sphere(cut_points)]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal', adjustable='box')
    offset = np.ones(3)
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

    # Draw a sphere.
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v)) + 1
    y = np.outer(np.sin(u), np.sin(v)) + 1
    z = np.outer(np.ones(np.size(u)), np.cos(v)) + 1
    ax.plot_wireframe(x, y, z, color='gray', alpha=0.1)

    ax_range = [0, 2]
    ax.set_xlim(ax_range)
    ax.set_ylim(ax_range)
    ax.set_zlim(ax_range)
    plt.show()