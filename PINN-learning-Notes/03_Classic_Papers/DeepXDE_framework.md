# PINN Framework: DeepXDE Implementation Guide

## 核心优势
DeepXDE 将 PINN 的构建过程模块化，使得研究者可以专注于物理方程本身，而非底层代码调试。

## 场景
求解 Burgers' Equation: 
$$u_t + u u_x - \frac{0.01}{\pi} u_{xx} = 0$$
Domain: $x \in [-1, 1], t \in [0, 1]$

## Python Implementation (DeepXDE)

```python
import deepxde as dde
import numpy as np

# 0. 准备工作：设置后端 (默认可能是 TensorFlow，也可以是 PyTorch)
# 在终端运行: export DDE_BACKEND=pytorch (推荐)

def pde(x, u):
    """
    Step 2: Define PDE
    x: 输入坐标 (x[:,0] 是 x, x[:,1] 是 t)
    u: 网络输出
    """
    # dde.grad.jacobian(u, x, i, j) 表示 du_i / dx_j
    # dde.grad.hessian(u, x, i, j) 表示 d^2u / dx_i dx_j
    
    du_x = dde.grad.jacobian(u, x, i=0, j=0)
    du_t = dde.grad.jacobian(u, x, i=0, j=1)
    du_xx = dde.grad.hessian(u, x, i=0, j=0)
    
    viscosity = 0.01 / np.pi
    
    # 返回 PDE 残差
    return du_t + u * du_x - viscosity * du_xx

def main():
    # --- Step 1: Geometry & Time Domain ---
    geom = dde.geometry.Interval(-1, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    # --- Step 3: BCs & ICs ---
    # Dirichlet BC: u(-1, t) = 0, u(1, t) = 0
    bc = dde.icbc.DirichletBC(
        geomtime, 
        lambda x: 0, 
        lambda x, on_boundary: on_boundary # 作用在所有空间边界上
    )
    
    # IC: u(x, 0) = -sin(pi * x)
    ic = dde.icbc.IC(
        geomtime, 
        lambda x: -np.sin(np.pi * x[:, 0:1]), 
        lambda x, on_initial: on_initial   # 作用在初始时刻 t=0
    )

    # 组合数据对象: 
    # num_domain: PDE 配点数量
    # num_boundary: 边界点数量
    # num_initial: 初始点数量
    data = dde.data.TimePDE(
        geomtime, 
        pde, 
        [bc, ic], 
        num_domain=2500, 
        num_boundary=100, 
        num_initial=160
    )

    # --- Step 4: Network Architecture ---
    # [2] -> [20, 20, 20] -> [1]
    # 输入层 2 (x, t), 3个隐藏层每层20个神经元, 输出层 1 (u)
    net = dde.nn.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")

    # --- Step 5: Model Compile & Train ---
    model = dde.Model(data, net)
    
    # 阶段 A: Adam
    model.compile("adam", lr=1e-3)
    model.train(iterations=10000)
    
    # 阶段 B: L-BFGS (一行代码实现切换！)
    model.compile("L-BFGS") 
    losshistory, train_state = model.train()
    
    # 可视化结果
    # dde.saveplot(losshistory, train_state, issave=True, isplot=True)

if __name__ == "__main__":
    main()