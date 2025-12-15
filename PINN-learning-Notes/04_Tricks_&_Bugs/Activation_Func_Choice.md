# PINN 中的激活函数选择 (Activation Functions in PINNs)

## 1. 核心问题：为什么不能用 ReLU？
在传统深度学习中，ReLU 是最常用的激活函数。但在 PINN (物理信息神经网络) 中，它通常是**不可用**的。

### 原因分析
PINN 的核心机制是利用自动微分 (Auto-grad) 计算偏导数，将其代入偏微分方程 (PDE) 计算残差 Loss。

$$Loss_{PDE} = || u_t - u_{xx} ||^2$$

* **一阶导数**：ReLU 的一阶导数是阶跃函数 (Step Function)。
* **二阶导数**：ReLU 的二阶导数在绝大多数点是 **0**，在原点是不可导的。

**结论**：如果 PDE 中包含二阶导项 (如热传导方程 $u_{xx}$ 或 波动方程)，使用 ReLU 会导致 $u_{xx}$ 恒为 0，物理 Loss 彻底失效，模型学不到任何物理规律。

## 2. 推荐的激活函数
PINN 需要**平滑 (Smooth)、无限可微**的激活函数。

### ✅ Tanh (双曲正切)
最经典的选择，Raissi 等人在 2019 年的开山之作中使用的就是 Tanh。
* **优点**：光滑，二阶导数非零。
* **代码**：`torch.nn.Tanh()`

### ✅ Sigmoid / Swish / SiLU
* Swish ($x \cdot \sigma(x)$) 在深层网络中表现有时优于 Tanh，因为它缓解了梯度消失问题。

### ✅ Sine (正弦函数)
在某些波动方程或高频震荡问题中（如 SIREN 网络），Sine 激活函数表现极佳。
* **注意**：初始化权重需要特殊技巧。

## 3. PyTorch 代码示例

\`\`\`python
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 20),
            nn.Tanh(),  # 关键点：使用 Tanh 而不是 ReLU
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )
\`\`\`