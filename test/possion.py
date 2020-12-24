import numpy as np
from scipy.sparse.linalg import spsolve
import pyamg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fealpy.pde.poisson_2d import CosCosData as PDE
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import DirichletBC 

from fealpy.tools.show import showmultirate, show_error_table

p = 2 # 有限元空间次数, 可以增大 p， 看输出结果的变化 
n = 4 # 初始网格加密次数 
maxit = 4 # 最大迭代次数  


pde = PDE()
mesh = pde.init_mesh(n=n)

errorType = ['$|| u - u_h||_{\Omega,0}$', # L2 误差
             '$||\\nabla u - \\nabla u_h||_{\Omega, 0}$' # H1 误差
             ]
errorMatrix = np.zeros((2, maxit), dtype=np.float)
NDof = np.zeros(maxit, dtype=np.float)

# 初始网格
fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)

for i in range(maxit):
    space = LagrangeFiniteElementSpace(mesh, p=p) # 建立有限元空间

    NDof[i] = space.number_of_global_dofs() # 有限元空间自由度的个数
    bc = DirichletBC(space, pde.dirichlet)  # DirichletBC 条件

    uh = space.function() # 有限元函数
    A = space.stiff_matrix() # 刚度矩阵
    F = space.source_vector(pde.source) # 载荷向量

    A, F = bc.apply(A, F, uh) # 处理边界条件 

    uh[:] = spsolve(A, F).reshape(-1) # 稀疏矩阵直接解法器

    #ml = pyamg.ruge_stuben_solver(A)  # 代数多重网格解法器
    #uh[:] = ml.solve(F, tol=1e-12, accel='cg').reshape(-1)

    errorMatrix[0, i] = space.integralalg.L2_error(pde.solution, uh) # 计算 L2 误差
    errorMatrix[1, i] = space.integralalg.L2_error(pde.gradient, uh.grad_value) # 计算 H1 误差

    if i < maxit-1:
        mesh.uniform_refine() # 一致加密网格

# 画函数图像
fig = plt.figure()
axes = fig.gca(projection='3d')
uh.add_plot(axes, cmap='rainbow')

# 画收敛阶图像
showmultirate(plt, 0, NDof, errorMatrix,  errorType, propsize=20)

# 输出 latex 误差表格
show_error_table(NDof, errorType, errorMatrix)
plt.show()
