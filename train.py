import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import os

# 从我们自己的文件中导入类
# 确保你已经修改了 modeling_tinyllm.py 文件以支持条件生成
from model import UNet
from noising_forward import NoisingForwardDiffusionScheduler

# --- 1. 设置超参数 ---
EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 28
IMG_CHANNELS = 1
TIMESTEPS = 1000

# --- 【新增】: 条件生成相关的超参数 ---
NUM_CLASSES = 10  # MNIST 数据集的类别数 (0-9)
CFG_PROB = 0.1  # Classifier-Free Guidance 的丢弃概率 (10%)


# --- 2. 准备数据 (无需修改) ---
def get_data_loader(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    data_root = "D:\\WhaleLearning\\Data"
    dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def train():
    print(f"开始训练，使用设备: {DEVICE}")

    # --- 3. 初始化核心组件 ---
    dataloader = get_data_loader(BATCH_SIZE)

    scheduler = NoisingForwardDiffusionScheduler(num_timesteps=TIMESTEPS, device=DEVICE)

    # 【修改】: 初始化模型时，传入 num_classes 参数
    model = UNet(
        in_channels=IMG_CHANNELS,
        out_channels=IMG_CHANNELS,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    checkpoints_dir = "checkpoints_conditional"  # 使用新的文件夹保存条件模型
    os.makedirs(checkpoints_dir, exist_ok=True)

    # --- 4. 训练主循环 ---
    for epoch in range(EPOCHS):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        # 【修改】: 从 dataloader 中同时获取图像和标签
        for step, (clean_images, labels) in enumerate(progress_bar):
            x_0 = clean_images.to(DEVICE)
            # 【新增】: 将标签也移动到指定设备
            labels = labels.to(DEVICE)

            t = torch.randint(0, TIMESTEPS, (x_0.shape[0],), device=DEVICE)
            x_t, true_noise = scheduler.add_noise(x_0, t)

            # --- 【新增】: Classifier-Free Guidance 核心逻辑 ---
            # 以 CFG_PROB 的概率，随机地将部分真实标签替换为“无条件标签”
            # 我们用 NUM_CLASSES (即 10) 作为无条件标签的 ID
            prob_mask = torch.rand(labels.shape[0], device=DEVICE) < CFG_PROB
            labels[prob_mask] = NUM_CLASSES  # 将掩码为 True 的位置替换为 10

            # --- 【修改】: 将加噪图片、时间步和处理后的标签一同送入模型 ---
            predicted_noise = model(x_t, t, labels)

            # d. 计算损失 (保持不变)
            loss = loss_fn(true_noise, predicted_noise)

            # e. 反向传播和优化 (保持不变)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix(loss=loss.item())

        # --- 5. 保存模型 ---
        model_path = os.path.join(checkpoints_dir, f"ddpm_mnist_cond_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Epoch {epoch + 1} 完成，模型已保存至 {model_path}")

    print("训练完成！")


if __name__ == '__main__':
    train()