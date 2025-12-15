# PINN Advanced: Adaptive Weights (Gradient Balancing)

## 核心思想
为了防止某个 Loss 项主导训练，我们需要动态调整权重 $\lambda$，使得 $\mathcal{L}_{data}$ 和 $\mathcal{L}_{PDE}$ 产生的梯度大小在一个数量级上。

## 算法流程 (简化版 Learning Rate Annealing)
1. 计算 Data Loss 和 PDE Loss。
2. 分别对网络每一层的权重求导，计算梯度的范数 (Norm)。
3. 根据梯度范数的比值更新 $\lambda$。
4. 使用更新后的 $\lambda$ 进行真正的反向传播更新参数。

## Python Implementation (PyTorch)

```python
import torch
import torch.nn as nn

class Adaptive_PINN:
    def __init__(self, model):
        self.model = model
        # 初始化权重 lambda 为 1.0 (这也是一个可训练参数或缓冲区)
        self.lambda_pde = torch.tensor(1.0, device='cpu')
        
        # 定义 alpha 用于平滑更新 (Moving Average)，防止权重跳变太剧烈
        self.alpha = 0.9 

    def get_grad_norm(self, loss, params):
        """
        计算特定 loss 对参数的梯度范数 (Gradient Norm)
        """
        # create_graph=True 是必须的，因为我们要对“梯度”做运算
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        
        # 将所有层的梯度拼接成一个长向量，并计算平均范数
        # 过滤掉 None (有些层可能不参与该 loss 的计算)
        grads_valid = [g.view(-1) for g in grads if g is not None]
        if not grads_valid:
            return torch.tensor(0.0)
            
        total_grad = torch.cat(grads_valid)
        return torch.mean(torch.abs(total_grad)) # 或者使用 torch.max

    def train_step_with_balance(self, optimizer, loss_data, loss_pde):
        """
        执行带梯度平衡的训练步
        """
        # 1. 获取网络中所有可训练参数
        # (通常只取权重的参数，忽略 bias，视具体论文实现而定)
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        # 2. 分别计算两个 Loss 的梯度范数
        # 注意：这里并不更新参数，只是为了“看”一下梯度有多大
        with torch.no_grad(): # 计算比率本身不需要梯度传播
             # 为了拿到梯度值，我们还是得 autograd，但在计算 lambda 时不需要追踪 lambda 的梯度
             pass 
        
        # 实际上 PyTorch 中需要这样操作：
        g_data = self.get_grad_norm(loss_data, params)
        g_pde = self.get_grad_norm(loss_pde, params)
        
        # 3. 计算新的权重 lambda_hat
        # 避免除以 0
        if g_pde == 0:
            lambda_hat = torch.tensor(1.0)
        else:
            lambda_hat = g_data / g_pde
        
        # 4. 平滑更新 lambda (Moving Average)
        # lambda_new = (1 - alpha) * lambda_old + alpha * lambda_hat
        self.lambda_pde = (1 - self.alpha) * self.lambda_pde + self.alpha * lambda_hat.item()
        
        # 5. 构建最终 Loss 并更新
        total_loss = loss_data + self.lambda_pde * loss_pde
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return total_loss.item(), self.lambda_pde