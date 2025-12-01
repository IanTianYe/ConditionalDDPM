import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms


class NoisingForwardDiffusionScheduler:
    """
    一个基于 alpha_t 表述的前向扩散过程调度器。

    该实现严格遵循以下单步转移公式:
    q(x_t | x_{t-1}) = N(x_t; sqrt(alpha_t) * x_{t-1}, (1 - alpha_t) * I)

    并利用其推导出的闭式解进行高效加噪:
    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
    """

    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cpu'):
        """
        初始化调度器并预计算所有必要的 alpha 系数。

        Args:
            num_timesteps (int): 总的扩散步数 T。
            beta_start (float): 噪声方差 beta 的初始值 (beta_t = 1 - alpha_t)。
            beta_end (float): 噪声方差 beta 的最终值。
            device (str or torch.device): 计算设备。
        """
        self.num_timesteps = num_timesteps
        self.device = device

        # 1. 定义 beta 调度 (variance schedule)
        # beta_t = 1 - alpha_t
        # 我们通过定义 beta 的线性增长来间接定义 alpha_t 的变化
        betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)

        # 2. 计算 alpha_t 参数
        # alpha_t 是核心，代表信号的保留率
        # 刚开始 beta 很小, alpha 很大代表着初始加噪过程极大地保留原始均值, 而方差很小, 之后逐步变大, 最后均值保留很小, 噪声很大, 贴近标准高斯分布
        # 例：self.alphas = tensor([0.9, 0.8, 0.7, 0.6, 0.5])
        self.alphas = 1.0 - betas

        # 3. 计算 alpha_bar_t (alphas_cumprod)
        # alpha_bar_t = product_{i=1 to t} (alpha_i)
        # 这是从 x_0 直接到 x_t 的闭式解中的关键部分
        # 例：tensor([0.9, 0.72, 0.504, 0.3024, 0.1512])
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0) # 累乘

        # 4. 预计算闭式解公式中需要用到的系数，方便直接调用
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod) # sqrt(alpha_bar_t)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod) # sqrt(1 - alpha_bar_t)

    def add_noise(self, x_0, t):
        """
        使用闭式解 q(x_t | x_0) 高效地为原始图像 x_0 添加噪声。
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon

        Args:
            x_0 (torch.Tensor): 原始清晰图像，形状为 (B, C, H, W) (Batch Size, Channels, Height, Width)。
            t (torch.Tensor): 一维张量，包含了批次中每个样本的目标时间步，形状为 (B,)。

        Returns:
            torch.Tensor: 加噪后的图像 x_t。
            torch.Tensor: 添加的噪声 epsilon (在训练中是模型的预测目标)。
        """
        # 确保输入在正确的设备上
        x_0 = x_0.to(self.device)
        t = t.to(self.device)

        # 从标准正态分布中采样噪声
        noise = torch.randn_like(x_0, device=self.device) # 维度和x_0相同，同时直接进行高斯噪声采样

        # 根据时间步 t, 从预计算的张量中提取对应的 sqrt(alpha_bar_t)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod.gather(dim=0, index=t) # 按时间步t直接在张量的第0的维度上获取元素

        # 提取对应的 sqrt(1 - alpha_bar_t)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod.gather(dim=0, index=t)

        # 为了进行广播运算，将系数的形状从 (B,) 调整为 (B, 1, 1, 1)
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(-1, 1, 1, 1)

        # 应用闭式解公式
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

        return x_t, noise

if __name__ == '__main__':
    # --- 1. 配置参数 ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # 根据您的截图，设置 MNIST 数据集的根目录
    # 请确保这个路径在您的电脑上是正确的
    MNIST_DATA_ROOT = "D:\\WhaleLearning\\Data"
    TIMESTEPS = 1000

    print(f"使用的设备: {DEVICE}")
    print(f"从路径加载数据集: {MNIST_DATA_ROOT}")

    # --- 2. 加载数据集并选择一张图片 ---
    # 定义图像预处理：转换为 Tensor 并归一化到 [-1, 1] 范围
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将 PIL Image [0, 255] 转为 Tensor [0, 1]
        transforms.Normalize(mean=[0.5], std=[0.5])  # 将 [0, 1] 范围归一化到 [-1, 1]
    ])

    # 加载训练数据集 (因为您有数据，所以 download=False)
    # 如果路径不正确或文件损坏，程序可能会报错
    try:
        train_dataset = torchvision.datasets.MNIST(
            root=MNIST_DATA_ROOT,
            train=True,
            download=False,
            transform=transform
        )
    except Exception as e:
        print(f"加载数据集失败! 请检查路径 '{MNIST_DATA_ROOT}' 是否正确。")
        print(f"错误信息: {e}")
        exit()  # 如果无法加载数据，则退出程序

    # 随机选择一张图片
    img_index = np.random.randint(0, len(train_dataset))
    x_0, label = train_dataset[img_index]

    # 为图片增加一个批次维度 (B=1)，使其形状变为 (B, C, H, W) -> (1, 1, 28, 28)
    x_0 = x_0.unsqueeze(0)

    print(f"成功加载数据集。共 {len(train_dataset)} 张训练图片。")
    print(f"已随机选择第 {img_index} 张图片 (标签: {label})。")
    print(f"输入张量的形状 (B, C, H, W): {x_0.shape}")

    # --- 3. 初始化调度器 ---
    scheduler = NoisingForwardDiffusionScheduler(num_timesteps=TIMESTEPS, device=DEVICE)

    # --- 4. 可视化前向扩散过程 ---

    # 定义要展示的时间步数量和具体的时间步
    num_images_to_show = 10
    display_timesteps = np.linspace(0, TIMESTEPS - 1, num_images_to_show, dtype=int)

    # 创建一个画布来展示图片
    fig, axes = plt.subplots(1, num_images_to_show + 1, figsize=(18, 3))


    # 封装一个函数用于将 Tensor 转为可显示的图像格式
    def postprocess(tensor):
        # 反归一化到 [0, 1]，移除批次和通道维度，并移至 CPU
        return tensor.mul(0.5).add(0.5).squeeze().cpu()


    # 首先，在第一个位置显示原始图像
    axes[0].imshow(postprocess(x_0), cmap='gray')
    axes[0].set_title(f"Original (t=0)")
    axes[0].axis('off')

    # 循环遍历定义好的时间步，进行加噪并显示
    for i, t_val in enumerate(display_timesteps):
        # 创建一个形状为 (B,) 的时间步张量，B=1
        t = torch.tensor([t_val], device=DEVICE, dtype=torch.long)

        # 调用核心函数进行加噪
        x_t, noise = scheduler.add_noise(x_0, t)

        # 在对应的子图上显示加噪后的图片
        ax = axes[i + 1]
        ax.imshow(postprocess(x_t), cmap='gray')
        ax.set_title(f"t={t_val}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()