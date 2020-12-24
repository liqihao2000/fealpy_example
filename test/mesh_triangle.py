import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import TriangleMesh

node = np.array([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0),],dtype = np.float)
cell = np.array([(1, 2, 0), (3, 0, 2)], dtype = np.int)

mesh = TriangleMesh(node, cell)
mesh.uniform_refine(n=3)

aa = np.float32(3.0)
print(type(aa))

node = mesh.entity('node')
edge = mesh.entity('edge')
cell = mesh.entity('cell')

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, showindex = True)
mesh.find_cell(axes, showindex = True)
plt.show()