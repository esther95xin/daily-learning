# PINN Math Foundations: Loss Function Formulation

## 核心概念
PINN 的损失函数将物理约束作为正则化项加入。
Total Loss = Data Mismatch (Supervised) + PDE Residual (Unsupervised).

## 场景
求解 Burgers' Equation: $u_t + u u_x - \nu u_{xx} = 0$
其中 $\nu = 0.01/\pi$.

## Python Implementation (PyTorch)

```python
import torch
import torch.nn as nn

class PINN_Loss_Example:
    def __init__(self, model, nu=0.01/torch.pi):
        self.model = model
        self.nu = nu
        self.loss_fn = nn.MSELoss()

    def compute_pde_residual(self, x, t):
        """
        计算 PDE 的残差。
        注意：x, t 必须是 requires_grad=True 的张量
        """
        # 1. 前向传播得到预测值 u
        # 拼接 x 和 t 作为网络输入
        u = self.model(torch.cat([x, t], dim=1))
        
        # 2. 利用自动微分计算导数
        # du/dt
        u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
        # du/dx
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        # d^2u/dx^2 (需要对 u_x 再次求导)
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
        
        # 3. 构造 PDE 残差 (Burgers' Eq)
        # f = u_t + u*u_x - nu*u_xx
        f = u_t + u * u_x - self.nu * u_xx
        
        return f

    def total_loss(self, x_data, t_data, u_data, x_collocation, t_collocation):
        """
        x_data, t_data, u_data: 有标签的观测数据 (用于 Data Loss)
        x_collocation, t_collocation: 随机采样的配点 (用于 PDE Loss)
        """
        
        # --- Part 1: Data Loss (监督学习) ---
        u_pred_data = self.model(torch.cat([x_data, t_data], dim=1))
        loss_data = self.loss_fn(u_pred_data, u_data)
        
        # --- Part 2: PDE Loss (物理约束) ---
        # 确保配点开启梯度追踪
        x_collocation.requires_grad = True
        t_collocation.requires_grad = True
        
        # 计算残差
        f_pred = self.compute_pde_residual(x_collocation, t_collocation)
        
        # 我们希望残差趋近于 0
        target_zeros = torch.zeros_like(f_pred)
        loss_physics = self.loss_fn(f_pred, target_zeros)
        
        # --- Total Loss ---
        # 这里的权重 1.0 可以根据训练情况调整
        loss = loss_data + loss_physics
        
        return loss, loss_data, loss_physics