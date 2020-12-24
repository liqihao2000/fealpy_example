
import sys
import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh import MeshFactory
from fealpy.functionspace import LagrangeFiniteElementSpace

# p = int(sys.argv[1])
p = 3

mf =  MeshFactory()

box = [0, 1, 0, 1]
mesh = mf.boxmesh2d(box, nx=1, ny=1, meshtype='tri')

space = LagrangeFiniteElementSpace(mesh,p=p)

ldof = space.number_of_local_dofs()
gdof = space.number_of_global_dofs()

bc = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],], dtype = np.float64)
phi = space.basis(bc)
gphi = space.grad_basis(bc)


print('ldof:',ldof)
print('gdof:',gdof)
print('bc:', bc.shape)
print('phi:', phi.shape)
print('pghi:', gphi.shape)

ipoints = space.interpolation_points()
cell2dof = space.cell_to_dof()
print('cell2dof:')
for i, val in enumerate(cell2dof):
	print(i,': ', val)


fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, node=ipoints, showindex=True, color='r', fontsize=24)
# mesh.find_cell(axes, showindex=True)
plt.show()


