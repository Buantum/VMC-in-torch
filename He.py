import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 确保使用GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

class TrialWaveFunction(nn.Module):
    def __init__(self):
        super(TrialWaveFunction, self).__init__()
        # 定义变分参数的对数，以确保它们始终为正
        self.alpha = nn.Parameter(torch.tensor(0.1574, device=device))           # 初始 alpha = exp(0) = 1.0
        self.beta = nn.Parameter(torch.tensor(0.0215, device=device))
        #self.zeta =  nn.Parameter(torch.tensor(1.7043, device=device))

    def log_psi(self, r1, r2):
        """
        计算波函数的对数
        r1, r2: [batch_size, 3]
        """
        r12 = torch.norm(r1 - r2, dim=1)  # [batch_size]
        log_slater = -1.79*(r1 + r2)
        log_jastrow = -self.alpha/(0.1 + self.beta*r12)
        return log_slater + log_jastrow

    def forward(self, r1, r2):
        """
        返回波函数值
        r1, r2: [batch_size, 3]
        """
        log_psi = self.log_psi(r1, r2)
        return torch.exp(log_psi)

def metropolis_sampling(wf, num_samples, step_size=0.2, burn_in=100, n_steps=100):
    """
    并行 Metropolis-Hastings 采样
    wf: 波函数模型
    num_samples: 采样数量
    step_size: 步长
    burn_in: 烧入步数
    n_steps: 每个采样链的采样步数
    返回: [num_samples, 3], [num_samples, 3] (两个电子的3D坐标)
    """
    # 初始化多个采样链，每个采样链包含两个电子的3D坐标
    # 使用高斯分布初始化，避免电子重叠
    initial_std = 1.0  # 初始标准差，可根据需要调整
    samples_r1 = torch.randn(num_samples, 3, device=device) * initial_std
    samples_r2 = torch.randn(num_samples, 3, device=device) * initial_std

    # 使用 no_grad 进行采样，以提高效率
    with torch.no_grad():
        # 计算初始波函数的对数值
        current_log_psi = wf.log_psi(samples_r1, samples_r2)  # [num_samples]

        # 烧入期
        for _ in range(burn_in):
            # 生成提议的坐标
            proposal_r1 = samples_r1 + torch.randn_like(samples_r1) * step_size
            proposal_r2 = samples_r2 + torch.randn_like(samples_r2) * step_size
            proposal_log_psi = wf.log_psi(proposal_r1, proposal_r2)  # [num_samples]

            # 计算接受率
            log_ratio = 2.0 * (proposal_log_psi - current_log_psi)  # log(|psi'|^2 / |psi|^2)
            acceptance_ratio = torch.exp(log_ratio).clamp(max=1.0)  # [num_samples]

            # 生成随机数以决定接受或拒绝
            rand = torch.rand(num_samples, device=device)
            accept = rand < acceptance_ratio  # [num_samples] bool

            # 更新采样点
            samples_r1[accept] = proposal_r1[accept]
            samples_r2[accept] = proposal_r2[accept]
            current_log_psi[accept] = proposal_log_psi[accept]

        # 进行采样步数
        total_accept = 0
        for _ in range(n_steps):
            # 生成提议的坐标
            proposal_r1 = samples_r1 + torch.randn_like(samples_r1) * step_size
            proposal_r2 = samples_r2 + torch.randn_like(samples_r2) * step_size
            proposal_log_psi = wf.log_psi(proposal_r1, proposal_r2)  # [num_samples]

            # 计算接受率
            log_ratio = 2.0 * (proposal_log_psi - current_log_psi)
            acceptance_ratio = torch.exp(log_ratio).clamp(max=1.0)  # [num_samples]

            # 生成随机数以决定接受或拒绝
            rand = torch.rand(num_samples, device=device)
            accept = rand < acceptance_ratio  # [num_samples] bool

            # 更新采样点
            samples_r1[accept] = proposal_r1[accept]
            samples_r2[accept] = proposal_r2[accept]
            current_log_psi[accept] = proposal_log_psi[accept]

            # 统计接受次数
            total_accept += accept.sum().item()

    # 计算总体接受率
    acceptance_rate = total_accept / (num_samples * n_steps)
    #print(f'Acceptance rate: {acceptance_rate:.2f}')

    return samples_r1, samples_r2

def local_energy(wf, r1, r2):
    """
    计算局部能量，使用自动微分
    wf: 波函数模型
    r1, r2: [batch_size, 3]
    返回: [batch_size]
    """
    # 确保 r1 和 r2 需要梯度
    r1 = r1.clone().detach().requires_grad_(True)
    r2 = r2.clone().detach().requires_grad_(True)

    psi = wf(r1, r2)  # [batch_size]

    # 计算梯度 ∇ψ
    grad_r1 = torch.autograd.grad(psi, r1, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
    grad_r2 = torch.autograd.grad(psi, r2, grad_outputs=torch.ones_like(psi), create_graph=True)[0]

    lap_r1 = torch.autograd.grad(grad_r1, r1, grad_outputs=torch.ones_like(grad_r1), create_graph=True)[0]
    lap_r2 = torch.autograd.grad(grad_r2, r2, grad_outputs=torch.ones_like(grad_r2), create_graph=True)[0]
    
    psi_safe = psi.clamp(min=1e-10)  # 防止除零
    laplacian = torch.sum(lap_r1, dim=-1) + torch.sum(lap_r2, dim=-1)
    kinetic = -0.5 * laplacian/ psi_safe  # [batch_size]

    # 势能部分
    r1_norm = torch.norm(r1, dim=1) + 1e-10  # [batch_size]
    r2_norm = torch.norm(r2, dim=1) + 1e-10  # [batch_size]
    r12 = torch.norm(r1 - r2, dim=1) + 1e-10  # [batch_size]

    potential = -2.0 / r1_norm - 2.0 / r2_norm + 1.0 / r12  # [batch_size]

    # 局部能量
    E_loc = kinetic + potential  # [batch_size]

    return E_loc

def optimize(wf, optimizer, num_steps=1000, num_samples=5000, step_size=0.5, burn_in=100, n_steps=100):
    for step in range(1, num_steps + 1):
        # 进行并行采样
        samples_r1, samples_r2 = metropolis_sampling(wf, num_samples, step_size=step_size, burn_in=burn_in, n_steps=n_steps)

        # 计算局部能量
        E_loc = local_energy(wf, samples_r1, samples_r2)  # [num_samples]
        E_mean = E_loc.mean()
        E_std = E_loc.std()

        # 计算 log Psi
        log_psi = wf.log_psi(samples_r1, samples_r2)  # [num_samples]

        # 定义损失函数，根据VMC的梯度公式
        # ∇θ <E> = 2 * <(E_loc - E_mean) * ∇θ log_psi>
        # 这样在反向传播时，梯度会正确计算
        delta_E = E_loc - E_mean 
        loss=torch.mean(delta_E.detach() * log_psi)

        # 清零之前的梯度
        optimizer.zero_grad()

        # 反向传播，计算梯度
        loss.backward()

        # 确认梯度正确连接
        if not log_psi.requires_grad:
            print(f'Warning: log_psi does not require grad at step {step}')

        # 更新参数
        optimizer.step()

        # 检查参数是否出现nan
        if torch.isnan(wf.alpha) or torch.isnan(wf.beta):
            print(f'Error: Parameters became NaN at step {step}')
            break

        # 打印能量和参数值
        if step % 100 == 0 or step == 1:
            print(f'Step {step}: <E> = {E_mean.item():.6f} ± {E_std.item()/torch.sqrt(torch.tensor(50000)):.6f} Ha, alpha = {wf.alpha.item():.4f}, beta = {wf.beta.item():.4f}')

def main():
    # 初始化波函数
    wf = TrialWaveFunction().to(device)

    # 选择优化器，降低学习率以防止发散
    optimizer = optim.Adam(wf.parameters(), lr=1e-4)

    # 开始优化
    optimize(wf, optimizer, num_steps=400, num_samples=50000, step_size=0.4, burn_in=20, n_steps=40)

    # 输出最终结果
    # 先进行采样，不需要梯度追踪
    with torch.no_grad():
        samples_r1, samples_r2 = metropolis_sampling(wf, num_samples=50000, step_size=0.4, burn_in=20, n_steps=40)

    # 计算局部能量，梯度追踪是必要的
    E_loc = local_energy(wf, samples_r1, samples_r2)
    E_mean = E_loc.mean().item()
    E_std = E_loc.std().item()

    print(f'Final Energy: {E_mean:.6f} ± {E_std/torch.sqrt(torch.tensor(50000, device=device)):.6f} Ha')
    print(f'Final alpha: {wf.alpha.item():.4f}, Final beta: {wf.beta.item():.4f}')

if __name__ == "__main__":
    main()
