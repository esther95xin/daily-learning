# PINN Math Foundations: Optimization Strategy (Adam + L-BFGS)

## 1. 概念定义 (Definition)

在 PINN 的训练中，单一的优化器往往难以同时满足收敛速度和高精度的要求。因此，标准的工业界/学术界做法是采用 **混合优化策略 (Hybrid Optimization Strategy)**。

* **Adam (Adaptive Moment Estimation):** 一阶优化算法。基于梯度的动量进行更新。
    * **角色：** “探路者”。负责在训练初期快速下降，找到全局最优解的大致区域。
* **L-BFGS (Limited-memory BFGS):** 拟牛顿法 (Quasi-Newton)。利用梯度的历史信息近似海森矩阵 (Hessian Matrix，二阶导数)。
    * **角色：** “狙击手”。负责在训练后期利用曲率信息，进行极高精度的收敛 (Fine-tuning)。

## 2. 工作原理 (Mechanism)

### 为什么需要混合策略？
PINN 的损失函数地形 (Loss Landscape) 往往非常复杂，呈现出“狭长的山谷”形状。

* **Adam 的局限：** 就像一个低头走路的徒步者，只看脚下的坡度。在狭长山谷中，容易在两壁之间来回震荡 (Zig-zag)，难以精确走到谷底中心。
* **L-BFGS 的优势：** 就像一个能感知地形全貌的滑雪者。通过近似二阶导数，它能感知地形的**曲率**，从而调整方向顺着山谷底部直接滑下去。

### 关键差异
* **Adam:** 每次迭代只计算一次 Loss 和梯度。
* **L-BFGS:** 为了寻找最佳步长，会在一步更新中**多次计算** Loss (Line Search)。因此，代码实现时需要传入一个 `closure` 函数。

## 3. 具体示例 (Example)

假设我们在训练 Burgers' Equation ($Loss_{Total} = Loss_{Data} + Loss_{PDE}$)。

* **阶段一 (Warm-up):** 使用 Adam 训练 5000 Epochs。
    * *现象：* Loss 从 $1.0$ 迅速下降到 $1.0 \times 10^{-2}$，然后开始震荡，无法继续下降。
    * *目的：* 避免 L-BFGS 在初期因地形太复杂而陷入错误的局部极小值。
* **阶段二 (Fine-tuning):** 切换为 L-BFGS。
    * *现象：* 仅仅几十步迭代，Loss 瞬间从 $10^{-2}$ 暴跌至 $10^{-6}$ 甚至更低。
    * *目的：* 满足物理方程对精度的严苛要求。

## 4. 代码实现 (PyTorch Implementation)

```python
import torch
import torch.nn as nn
import torch.optim as optim

class PINN_Optimizer_Example:
    def __init__(self, model):
        self.model = model
    
    def calculate_loss(self):
        """
        这里封装了 Data Loss + PDE Loss 的计算逻辑
        返回一个 Tensor scalar
        """
        # 伪代码：实际项目中这里会调用你的 loss 计算函数
        # loss = compute_data_loss() + compute_pde_loss()
        return torch.tensor(0.0, requires_grad=True) 

    def train(self, steps_adam=5000, steps_lbfgs=1000):
        print("--- Phase 1: Adam Optimization (Warm-up) ---")
        # Adam 不需要 closure，适合快速迭代
        optimizer_adam = optim.Adam(self.model.parameters(), lr=1e-3)
        
        for step in range(steps_adam):
            optimizer_adam.zero_grad()
            loss = self.calculate_loss()
            loss.backward()
            optimizer_adam.step()
            
            if step % 1000 == 0:
                print(f"Adam Step {step}, Loss: {loss.item():.5f}")

        print("--- Phase 2: L-BFGS Optimization (Fine-tuning) ---")
        # L-BFGS 参数配置 (标准配置)
        optimizer_lbfgs = optim.LBFGS(
            self.model.parameters(), 
            lr=1.0,                 # L-BFGS 的学习率通常设为 1.0
            max_iter=steps_lbfgs,   # 单次 step 内部的最大迭代次数
            max_eval=steps_lbfgs,   # 函数评估最大次数
            tolerance_grad=1e-7,    # 梯度容差
            tolerance_change=1e-9,  # 变化量容差
            history_size=100,       # 内存中保留的历史梯度数量
            line_search_fn="strong_wolfe" # 强沃尔夫线搜索 (关键！保证收敛稳定性)
        )

        # 定义 closure 函数：允许 L-BFGS 在内部多次重新评估 Loss
        def closure():
            optimizer_lbfgs.zero_grad()
            loss = self.calculate_loss()
            loss.backward()
            return loss

        # L-BFGS 的 step 只需要调用一次 (或者外层包裹少量循环)
        # 它会在这一步内自动进行 max_iter 次优化
        optimizer_lbfgs.step(closure)
        
        print("Training Finished.")