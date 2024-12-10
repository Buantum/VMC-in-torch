import torch
import torch.optim as optim

# 设置随机种子以确保结果可重复
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义变分参数 alpha
alpha = torch.tensor(0.5, requires_grad=True, device=device)  # 初始猜测

def hydrogen_wavefunction(r, alpha):
    """
    氢原子波函数 Ansatz：ψ(r) = α^(3/2) * exp(-α * r)
    参数:
        r (torch.Tensor): 位置向量 [num_particles, 3]
        alpha (torch.Tensor): 变分参数
    返回:
        torch.Tensor: 波函数值 [num_particles]
    """
    r_norm = torch.norm(r, dim=1)  # 计算每个粒子的距离 r = sqrt(x^2 + y^2 + z^2)
    return torch.exp(-alpha * r_norm)  # 返回波函数值 [num_particles]

def local_energy(r, alpha):
    """
    计算氢原子的局部能量。
    动能和势能项的总和。
    参数:
        r (torch.Tensor): 位置向量 [num_particles, 3]
        alpha (torch.Tensor): 变分参数
    返回:
        torch.Tensor: 局部能量 [num_particles]
    """
    # 计算波函数和梯度
    wavefunction = hydrogen_wavefunction(r, alpha)  # [num_particles]
    
    # 计算梯度 ∇ψ
    r.requires_grad_(True)
    psi = hydrogen_wavefunction(r, alpha)  # 计算波函数
    grad_psi = torch.autograd.grad(psi.sum(), r, create_graph=True)[0]  # [num_particles, 3]
    
    # 计算拉普拉斯算符 ∇^2ψ (二阶导数)
    grad2_psi = torch.autograd.grad(grad_psi.sum(), r, create_graph=True)[0]  # [num_particles, 3]
    laplacian_psi = grad2_psi.sum(dim=1)  # [num_particles]
    
    # 动能项：T = -0.5 * ∇^2ψ / ψ
    kinetic_energy = -0.5 * laplacian_psi / wavefunction
    
    # 势能项：V = -1 / r
    r_norm = 1e-10+torch.norm(r, dim=1)  # [num_particles]
    potential_energy = -1 / r_norm  # [num_particles]
    
    # 总能量
    total_energy = kinetic_energy + potential_energy
    return total_energy

def metropolis_sampling(a, num_particles, num_steps, step_size=0.5, burn_in=10):
    """
    使用并行的Metropolis-Hastings算法进行采样。

    参数:
        a (torch.Tensor): 变分参数a。
        num_particles (int): 系统中的粒子数量。
        num_steps (int): 每个粒子要采样的迭代次数。
        step_size (float): 提议步长。
        burn_in (int): Burn-in阶段的步数。

    返回:
        torch.Tensor: 形状为 [num_steps * num_particles, 3] 的采样点。
    """
    samples = []
    accept = 0
    total = 0

    # 初始位置: [num_particles, 3]
    r = torch.zeros(num_particles, 3, device=device)  # 多个粒子的初始点 (x, y, z)
    #r=torch.randn_like(r)
    psi = torch.exp(-a * torch.norm(r, dim=1))    # ψ(r) = e^{-a r}

    # Burn-in阶段
    for _ in range(burn_in):
        # 提议新位置
        r_new = 5*torch.randn_like(r)   # [num_particles, 3]
        psi_new = hydrogen_wavefunction(r_new, a)**2  # [num_particles]

        lognew=torch.log(hydrogen_wavefunction(r_new, a))
        logold=torch.log(hydrogen_wavefunction(r, a))

        newby=torch.exp(2*(lognew-logold))



        # 计算接受概率 α = min(1, e^{-2a (r_new - r)})
        #newby=(hydrogen_wavefunction(r_new, a)/hydrogen_wavefunction(r, a))**2 # [num_particles]
        alpha = torch.minimum(torch.ones_like(newby), torch.exp(newby))        # [num_particles]

        # 生成随机数并决定是否接受
        rand = torch.rand(num_particles, device=device)
        accept_mask = rand < alpha  # [num_particles]

        # 更新接受的粒子的位置和ψ值
        r[accept_mask] = r_new[accept_mask]
        psi[accept_mask] = psi_new[accept_mask]
        accept += accept_mask.sum().item()
        total += num_particles

    acceptance_rate = accept / total
    #print(f"Burn-in阶段接受率: {acceptance_rate:.2f}")

    # 采样阶段
    accept = 0
    total = 0
    for _ in range(num_steps):
        # 提议新位置
        r_new = 5*torch.randn_like(r)      # [num_particles, 3]
        psi_new = hydrogen_wavefunction(r_new, a)**2 # [num_particles]

        lognew=torch.log(hydrogen_wavefunction(r_new, a))
        logold=torch.log(hydrogen_wavefunction(r, a))

        newby=torch.exp(2*(lognew-logold))

        #newby=(hydrogen_wavefunction(r_new, a)/hydrogen_wavefunction(r, a))**2
        
        alpha = torch.minimum(torch.ones_like(newby), newby)        # [num_particles]

        # 生成随机数并决定是否接受
        rand = torch.rand(num_particles, device=device)
        accept_mask = rand < alpha  # [num_particles]

        # 更新接受的粒子的位置和ψ值
        r[accept_mask] = r_new[accept_mask]
        psi[accept_mask] = psi_new[accept_mask]
        accept += accept_mask.sum().item()
        total += num_particles

        # 记录当前样本
        samples.append(r.clone())

    acceptance_rate = accept / total
    print(f"采样阶段接受率: {acceptance_rate:.2f}")

    # 将所有样本堆叠成一个张量，并调整形状为 [num_steps * num_particles, 3]
    samples = torch.stack(samples)        # [num_steps, num_particles, 3]
    samples = samples.view(-1, 3)         # [num_steps * num_particles, 3]
    return samples                       # [num_steps * num_particles, 3]

import matplotlib.pyplot as plt
def train_vmc(num_iterations=1000, num_samples=10000, num_particles=100, step_size=0.5, learning_rate=0.01, save_path="vmc_training_plot.png"):
    """
    训练VMC模型以优化参数alpha。

    参数:
        num_iterations (int): 训练迭代次数
        num_samples (int): 每次迭代中每个粒子要采样的样本数量
        num_particles (int): 系统中的粒子数量
        step_size (float): 提议步长
        learning_rate (float): 优化器的学习率
        save_path (str): 图像保存路径
    """
    optimizer = optim.Adam([alpha], lr=learning_rate)

    # 创建列表记录每次迭代的E_mean和标准差
    E_mean_list = []
    std_list = []
    iteration_list = []

    for it in range(1, num_iterations + 1):
        # 采样
        samples = metropolis_sampling(alpha, num_particles, num_steps=num_samples, step_size=step_size)

        # 计算局部能量
        E_loc = local_energy(samples, alpha)  # [num_samples * num_particles]

        # 计算能量期望值
        E_mean = E_loc.mean()

        # 计算标准差
        E_std = E_loc.std()

        # 每次迭代记录一次        std_list.append(E_std.item() / torch.sqrt(torch.tensor(float(num_samples))))
        E_mean_list.append(E_mean.item())
        std_list.append(E_std.item() / torch.sqrt(torch.tensor(float(num_particles))))
        iteration_list.append(it)

        # 计算梯度 ∇α <E> = 2 <(E_loc - <E>) * (-r)>
        #r_abs = torch.norm(samples, dim=1)  # [num_samples * num_particles] 计算每个粒子的绝对值
        delta_E = E_loc - E_mean            # [num_samples * num_particles]
        wavefunction = hydrogen_wavefunction(samples, alpha)
        lnwavefunction = torch.log(wavefunction)

        loss = torch.mean(delta_E.detach() *lnwavefunction ) 
        optimizer.zero_grad()

        loss.backward()

        # 更新参数
        optimizer.step()
        

        # 每100次迭代打印一次
        if it % 100 == 0 or it == 1:
            print(f"迭代 {it}: <E> = {E_mean.item():.6f}, 标准差 = {E_std.item()/torch.sqrt(torch.tensor(num_particles)):.6f}, alpha = {alpha.item():.6f}")

    print(f"优化完成: <E> = {E_mean.item():.6f}, 最优 alpha = {alpha.item():.6f}")

    # 仅选择每100次迭代的数据点绘制
    selected_iterations = iteration_list[::5]  # 每100次选择一个数据点
    selected_E_mean = E_mean_list[::5]         # 每100次选择一个能量期望值
    selected_std = std_list[::5]               # 每100次选择一个标准差

    # 绘制能量期望值和标准差（作为误差条）随迭代次数变化的图像
    plt.figure(figsize=(10, 6))

    # 使用误差条绘制能量期望值和标准差
    plt.errorbar(selected_iterations, selected_E_mean, yerr=selected_std, fmt='-o', color='blue', label='Energy Expectation', capsize=5)

    plt.xlabel('Iteration')
    plt.ylabel('<E>')
    plt.title('Energy Expectation with Error Bars (Every 100 Iterations)')
    plt.grid(True)
    plt.legend()
        # 保存图像到文件
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"图像已保存到: {save_path}")
    plt.close()  # 关闭图形以释放内存

# 运行训练
if __name__ == "__main__":
    # 参数设置
    import time
    num_iterations = 400      # 训练迭代次数
    num_samples = 1         # 每次迭代中每个粒子采样的样本数量
    num_particles = 100000         # 系统中的粒子数量
    step_size = 0.100            # 提议步长
    learning_rate = 0.01      # 优化器的学习率

    time_start = time.time()
    train_vmc(num_iterations=num_iterations, num_samples=num_samples, num_particles=num_particles,
              step_size=step_size, learning_rate=learning_rate)
    time_end = time.time()
    print(f"训练完成，总用时: {time_end - time_start:.2f} 秒")
