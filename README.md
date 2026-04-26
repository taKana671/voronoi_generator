# voronoi_generator

A submodule that uses SciPy's `scipy.spatial.Voronoi` to compute 2D or 3D Voronoi partitions and generate the vertex coordinates of the Voronoi regions.

# Requirements

* numpy 2.2.6
* scipy 1.16.2
* shapely 2.1.2
* matplotlib 3.10.7
  
# Environment

* Python 3.13
* Windows11

# Modules

## voronoi_2d

Using `SciPy` and `Shapely`, generate a Voronoi diagram, then clip the 2D Voronoi regions to a square (or arbitrary rectangle) area.<br>
In [VoronoiCity2](https://github.com/taKana671/VoronoiCity2), 3D models are generated from Voronoi polygons to create a city.<br>
The visualization results are as follows.

<img width="865" height="360" alt="Image" src="https://github.com/user-attachments/assets/ff73ca15-8932-418a-b2ea-9b12388fab76" />

<pre>
  def visualize1():
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for region in BoundedVoronoiGenerator(cnt_points=20, shrink=0.):
        ax.add_patch(Polygon(region, facecolor=np.random.uniform(0, 1, 3), alpha=0.6, edgecolor='gray'))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.show()


  def visualize2():
    fig = plt.figure()
    ax = fig.add_subplot(111)

    polygon = np.array([
        [0.57122648, 0.26608489],
        [0.37531127, 0.42911859],
        [0.45373198, 0.73934282],
        [0.64410673, 0.67292705],
        [0.62157049, 0.3502432]
    ])

    pts = [pt for pt in ConvexPolygonGenerator(polygon)]
    for region in BoundedVoronoiGenerator(pts=pts, bnd=polygon, shrink=0):
        ax.add_patch(Polygon(region, facecolor=np.random.uniform(0, 1, 3), alpha=0.6, edgecolor='gray'))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.show()
</pre>


## voronoi_3d

### clip2cube

Perform a 3D Voronoi partition by using SciPy's `scipy.spatial.Voronoi`, calculate the vertex coordinates of the Voronoi regions (polyhedra), and clip them to a cube.<br>
In [Clipped3DVoronoi](https://github.com/taKana671/Clipped3DVoronoi), I created 3D models of polyhedrons from vertex coordinates that have been clipped to a cube.<br>
As shown below, it is also possible to visualize a 3D Voronoi diagram.

<img width="572" height="396" alt="Image" src="https://github.com/user-attachments/assets/588c43e6-ba12-4dfd-9dcd-2b5e937a3987" />

<br>

<pre>
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from voronoi_generator.voronoi_3d.clip2cube.clip2cube import VoronoiClipped2Cube


def visualize(cut_points=30, cube_size=1., diff=.5, alpha=.6):
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
</pre>

