import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# matplotlib inline

from fealpy.mesh import MeshFactory

mf = MeshFactory()

box = [0, 1, 0, 1]
mesh = mf.boxmesh2d(box, nx = 4, ny = 4, meshtype = 'tri')
fig = plt.figure()
axes  =fig.gca()
mesh.add_plot(axes)
plt.show()