import torch
from torchvision.utils import make_grid
from tqdm import tqdm
import matplotlib.pyplot as plt

# 从我们自己的文件中导入类
# 确保你已经修改了 modeling_tinyllm.py 文件
from model import UNet
from noising_forward import NoisingForwardDiffusionScheduler

# --- 1. 设置采样参数 ---
# 【修改】: 更新检查点路径为条件模型的路径
CHECKPOINT_PATH = "checkpoints_conditional/ddpm_mnist_cond_epoch_20.pth"
NUM_IMAGES = 64
TIMESTEPS = 800  # 必须与训练时使用的 TIMESTEPS 保持一致
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 28
IMG_CHANNELS = 1

# --- 【新增】: 条件生成相关的参数 ---
NUM_CLASSES = 10
GUIDANCE_SCALE = 7.0  # 引导强度 w。通常设置在 5 到 10 之间效果较好
TARGET_CLASS = 7  # 你想生成的数字 (0-9)


def sample():
    print(f"开始采样，使用设备: {DEVICE}")
    print(f"目标类别: {TARGET_CLASS}, 引导强度: {GUIDANCE_SCALE}")

    # --- 2. 初始化核心组件 ---
    scheduler = NoisingForwardDiffusionScheduler(num_timesteps=TIMESTEPS, device=DEVICE)

    # 【修改】: 初始化模型时，必须传入 num_classes 参数
    # 注意：这里的 num_classes 仍然是 10，因为模型内部的 Embedding 层会自动处理 +1 的情况
    model = UNet(
        in_channels=IMG_CHANNELS,
        out_channels=IMG_CHANNELS,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    print(f"正在从 '{CHECKPOINT_PATH}' 加载模型权重...")
    try:
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print(f"错误: 找不到模型文件 '{CHECKPOINT_PATH}'。请确认路径和文件名是否正确。")
        return

    model.eval()

    # --- 3. 执行反向去噪过程 ---
    current_x = torch.randn(NUM_IMAGES, IMG_CHANNELS, IMG_SIZE, IMG_SIZE).to(DEVICE)
    progress_bar = tqdm(reversed(range(TIMESTEPS)), desc="Sampling", total=TIMESTEPS)

    # --- 【新增】: 准备条件生成所需的标签 ---
    # 1. 目标标签：所有生成的图片都使用同一个目标类别
    target_labels = torch.full((NUM_IMAGES,), TARGET_CLASS, device=DEVICE, dtype=torch.long)
    # 2. 无条件标签：使用我们约定的 ID (NUM_CLASSES, 即 10) 作为无条件生成的标签
    uncond_labels = torch.full((NUM_IMAGES,), NUM_CLASSES, device=DEVICE, dtype=torch.long)

    with torch.no_grad():
        for t in progress_bar:
            t_tensor = torch.full((NUM_IMAGES,), t, device=DEVICE, dtype=torch.long)

            # --- 【核心修改】: 实现 Classifier-Free Guidance ---
            # 1. 预测条件噪声 (传入目标标签)
            cond_noise = model(current_x, t_tensor, target_labels)

            # 2. 预测无条件噪声 (传入无条件标签)
            uncond_noise = model(current_x, t_tensor, uncond_labels)

            # 3. 使用引导公式组合两个预测
            # predicted_noise = 无条件噪声 + w * (条件噪声 - 无条件噪声)
            predicted_noise = uncond_noise + GUIDANCE_SCALE * (cond_noise - uncond_noise)
            # --- 核心修改结束 ---

            # 后续的去噪步骤与之前完全相同，只是现在使用了引导后的 predicted_noise
            alpha_t = scheduler.alphas[t]
            # 【修正】: 此处应使用 scheduler.alphas_cumprod
            alpha_bar_t = scheduler.alphas_cumprod[t]
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)

            coeff = (1 - alpha_t) / sqrt_one_minus_alpha_bar_t
            model_mean = (1 / torch.sqrt(alpha_t)) * (current_x - coeff * predicted_noise)

            if t > 0:
                variance = 1 - alpha_t
                noise = torch.randn_like(current_x)
                current_x = model_mean + torch.sqrt(variance) * noise
            else:
                current_x = model_mean

    # --- 4. 后处理并直接显示图片 ---
    generated_images = (current_x.clamp(-1, 1) + 1) / 2  # clamp 确保数值范围
    grid = make_grid(generated_images, nrow=int(NUM_IMAGES ** 0.5), padding=2)

    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).cpu())
    # 【修改】: 更新标题以反映条件生成
    plt.title(f"Generated MNIST Samples for Digit '{TARGET_CLASS}'")
    plt.axis('off')
    plt.show()

    print(f"\n采样完成！已在窗口中显示 {NUM_IMAGES} 张生成的图片。")


if __name__ == '__main__':
    sample()