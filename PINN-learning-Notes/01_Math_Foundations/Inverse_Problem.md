# PINN 基础 (4): 逆问题与参数发现 (Inverse Problems)

## 1. 概念定义 (Definition)

**逆问题 (Inverse Problem)** 指的是在已知物理方程结构但缺失部分物理参数（如粘度、扩散率、波速）的情况下，利用稀疏的观测数据，反向推断出这些未知参数，并同时重构全场解的过程。

在 PINN 的语境下，逆问题不再需要传统数值方法（如 FEM）中昂贵的“外层优化-内层求解”循环，而是通过将未知参数视为**可训练变量**，实现了参数发现与方程求解的**联合优化 (Joint Optimization)**。

## 2. 工作原理 (Mechanism)

PINN 处理逆问题的核心机制在于**“变量化”**与**“梯度回传”**。

1.  **变量化 (Variabilization):** 在计算图中，未知的物理参数（记为 $\lambda$）不再是常数，而被初始化为一个**可训练张量 (Trainable Tensor)**，其地位等同于神经网络的权重 $W$。
2.  **损失耦合 (Loss Coupling):** 损失函数同时包含数据误差和物理残差：
    $$\mathcal{L}_{total} = \mathcal{L}_{data}(u_{obs}, u_{pred}) + \mathcal{L}_{PDE}(u_{pred}; \lambda)$$
3.  **联合梯度下降:** 优化器在反向传播时，同时计算两条路径的梯度：
    * $\partial \mathcal{L} / \partial W$: 更新网络权重以拟合观测数据。
    * $\partial \mathcal{L} / \partial \lambda$: 更新物理参数以最小化 PDE 残差。

> **直觉类比：** > 这就像**侦探破案**。
> * **正问题：** 已知凶手和凶器（已知参数），推演案发过程（求状态）。
> * **逆问题（PINN）：** 看到案发现场（观测数据），根据物理常识（PDE），反推凶手拿的凶器有多重（未知参数）。侦探（优化器）不断微调对凶器重量的猜测，直到推演出的过程与现场痕迹完美吻合。

## 3. 具体示例 (Example)

**场景：** 求解 Burgers' Equation，但流体的粘度系数 $C$ 未知。
$$u_t + u u_x - C u_{xx} = 0$$

* **已知信息：** 方程形式，以及在流场中随机采集的 $N$ 个点的流速观测值 $(x_i, t_i, u_i)$。
* **目标：** 训练神经网络拟合 $u(x,t)$，同时让网络自动算出 $C$ 的值（假设真值为 $0.01/\pi$）。

## 4. 代码实现 (DeepXDE Snippet)

```python
import deepxde as dde
import numpy as np

# --- 1. 定义未知参数为可训练变量 ---
# 给予一个初始猜测值 (例如 0.0)
C = dde.Variable(0.0)

def pde(x, u):
    """
    Step 2: 在 PDE 定义中直接使用变量 C
    """
    dy_t = dde.grad.jacobian(u, x, i=0, j=1)
    dy_x = dde.grad.jacobian(u, x, i=0, j=0)
    dy_xx = dde.grad.hessian(u, x, i=0, j=0)
    
    # PINN 会试图找到一个 C，使得这个残差为 0
    return dy_t + u * dy_x - C * dy_xx

def main():
    # 几何定义
    geom = dde.geometry.Interval(-1, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    # --- 3. 导入观测数据 (核心) ---
    # 这里模拟生成一些观测数据
    # 实际科研中，这里通常是 load_data_from_file()
    observe_x = np.random.uniform(-1, 1, (2000, 2)) # 包含 x 和 t
    # 假设我们知道这2000个点的真实值
    # (此处用解析解模拟真实值，用于生成标签)
    observe_y = -np.sin(np.pi * observe_x[:, 0:1]) * np.exp(-0.01/np.pi * np.pi**2 * observe_x[:, 1:2])
    
    # 将观测数据通过 PointSetBC 喂给模型
    bc = dde.icbc.PointSetBC(observe_x, observe_y)

    data = dde.data.TimePDE(
        geomtime, 
        pde, 
        [bc], # 逆问题主要依赖观测数据 BC
        num_domain=2000, 
        num_boundary=0, 
        num_initial=0,
        anchors=observe_x # 锚点：显式告诉模型这些点很重要
    )

    net = dde.nn.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")
    model = dde.Model(data, net)

    # --- 4. 注册外部变量 ---
    # external_trainable_variables 列表告诉 DeepXDE 还有谁需要更新
    model.compile("adam", lr=1e-3, external_trainable_variables=[C])
    
    # 使用回调函数监控参数 C 的收敛曲线
    variable_callback = dde.callbacks.VariableValue(
        [C], period=500, filename="C_history.dat"
    )
    
    print("开始逆问题求解训练...")
    model.train(iterations=10000, callbacks=[variable_callback])
    
    print(f"预测出的粘度系数 C: {C.value}")
    print(f"理论真实值: {0.01/np.pi}")

if __name__ == "__main__":
    main()