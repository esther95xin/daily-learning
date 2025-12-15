# PINN 基础 Loss 函数设计

## 1. 原理公式
PINN 的总 Loss 通常由两部分组成：方程残差 ($Loss_{PDE}$) 和 边界条件 ($Loss_{BC}$)。

$$
Loss = \lambda_{PDE} Loss_{PDE} + \lambda_{BC} Loss_{BC}
$$

其中，$Loss_{PDE}$ 定义为：
$$
Loss_{PDE} = \frac{1}{N_f} \sum_{i=1}^{N_f} |f(x_f^i, t_f^i)|^2
$$

## 2. PyTorch 代码实现片段
这是我常用的一个基础 Loss 计算模块：

\`\`\`python
import torch

def physics_informed_loss(model, x, t, u_true):
    # 1. 开启梯度追踪
    x.requires_grad = True
    t.requires_grad = True
    
    # 2. 前向传播预测 u
    u_pred = model(torch.cat([x, t], dim=1))
    
    # 3. 自动微分求导 (关键步骤!)
    u_x = torch.autograd.grad(u_pred, x, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
    u_t = torch.autograd.grad(u_pred, t, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    
    # 4. 定义 PDE 残差 (以热传导方程为例: u_t - u_xx = 0)
    f_pred = u_t - u_xx 
    
    # 5. 计算 MSE Loss
    loss_pde = torch.mean(f_pred ** 2)
    loss_data = torch.mean((u_pred - u_true) ** 2)
    
    return loss_pde + loss_data
\`\`\`

## 3. 常见坑点 (Debug) ⚠️
* **二阶导数为0？**
    * **原因**：很可能你用了 `ReLU` 激活函数。ReLU 的二阶导数恒为 0，无法用于 PINN。
    * **解决**：必须换成 `Tanh`, `Sin` 或 `Swish` 这种光滑激活函数。
* **显存爆炸？**
    * **原因**：`create_graph=True` 会保存计算图，非常吃显存。
    * **解决**：确保只在需要求高阶导数的地方开这个选项。