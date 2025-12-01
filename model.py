import torch
import torch.nn as nn
import math


# SinusoidalPositionEmbeddings 类
class SinusoidalPositionEmbeddings(nn.Module):
    """
    将时间步 t 转换为高维向量表示。
    这个模块来自 Transformer 模型的经典位置编码。
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


# ResidualBlock 类
class ResidualBlock(nn.Module):
    """
    U-Net 中的基本构建块，包含两个卷积层、组归一化、激活函数以及时间嵌入的注入。
    """

    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        # 时间嵌入的投影层
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        # 第一个卷积块
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU()
        )

        # 第二个卷积块
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU()
        )

        # 残差连接的匹配层（如果输入输出通道数不同）
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(x)

        # 注入时间信息
        # t_emb [B, time_emb_dim] -> [B, out_channels] -> [B, out_channels, 1, 1]
        time_info = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + time_info

        h = self.conv2(h)

        # 添加残差连接
        return h + self.residual_conv(x)


class UNet(nn.Module):
    """
    一个简化的、支持条件生成的 U-Net 架构。
    它的任务是接收加噪图片 x_t、时间步 t 和类别标签 c，预测出添加的噪声。
    """

    def __init__(self, in_channels=1, out_channels=1, time_emb_dim=256, num_classes=10):
        super().__init__()

        # --- 时间步嵌入 ---
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        # --- 类别标签嵌入 ---
        # 我们需要能处理10个数字类别 (0-9) 和1个无条件类别。
        # 因此嵌入层的输入维度是 num_classes + 1。
        self.class_emb = nn.Embedding(num_classes + 1, time_emb_dim)

        # --- 编码器 (下采样部分) ---
        self.initial_conv = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)

        self.down1 = ResidualBlock(64, 128, time_emb_dim)
        self.pool1 = nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1)

        self.down2 = ResidualBlock(128, 256, time_emb_dim)
        self.pool2 = nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1)

        # --- 瓶颈部分 ---
        self.bottleneck = ResidualBlock(256, 256, time_emb_dim)

        # --- 解码器 (上采样部分) ---
        self.up1_conv_t = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        # 输入通道数: 128 (来自上采样) + 256 (来自跨层连接 x3) = 384
        self.up1 = ResidualBlock(384, 128, time_emb_dim)

        self.up2_conv_t = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        # 输入通道数: 64 (来自上采样) + 128 (来自跨层连接 x2) = 192
        self.up2 = ResidualBlock(192, 64, time_emb_dim)

        # --- 输出层 ---
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)

    #  forward 方法签名，新增 c (class label)
    def forward(self, x, t, c=None):
        # 1. 时间嵌入
        t_emb = self.time_mlp(t)

        # --- 类别嵌入与融合 ---
        # 如果提供了类别标签 c (在训练和条件采样时)
        if c is not None:
            # 将类别 ID 转换为嵌入向量
            class_emb = self.class_emb(c)
            # 将类别信息与时间信息融合
            t_emb = t_emb + class_emb
        # 如果 c is None (无条件采样时)，则 t_emb 只包含时间信息

        # 后续流程完全不变，因为 t_emb 已经包含了所需的所有条件信息
        # 2. 编码器
        x1 = self.initial_conv(x)

        x2 = self.down1(x1, t_emb)
        p1 = self.pool1(x2)

        x3 = self.down2(p1, t_emb)
        p2 = self.pool2(x3)

        # 3. 瓶颈
        b = self.bottleneck(p2, t_emb)

        # 4. 解码器
        up1 = self.up1_conv_t(b)
        concat1 = torch.cat([up1, x3], dim=1)
        x4 = self.up1(concat1, t_emb)

        up2 = self.up2_conv_t(x4)
        concat2 = torch.cat([up2, x2], dim=1)
        x5 = self.up2(concat2, t_emb)

        # 5. 输出
        return self.output(x5)


if __name__ == '__main__':
    # ---  更新测试，验证条件模型的输入输出 ---

    BATCH_SIZE = 16
    IMG_CHANNELS = 1
    IMG_SIZE = 28
    NUM_CLASSES = 10  # MNIST 类别数

    # 1. 创建一个条件 U-Net 实例
    model = UNet(
        in_channels=IMG_CHANNELS,
        out_channels=IMG_CHANNELS,
        num_classes=NUM_CLASSES
    )

    # 2. 创建假的输入张量
    dummy_x = torch.randn(BATCH_SIZE, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
    dummy_t = torch.randint(0, 1000, (BATCH_SIZE,))
    # 创建假的类别标签张量 (0-9)
    dummy_c = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))

    # 3. 执行前向传播 (传入类别标签)
    predicted_noise = model(dummy_x, dummy_t, dummy_c)

    # 4. 打印输入和输出的形状以进行验证
    print("条件 U-Net 模型测试")
    print(f"输入图像形状: {dummy_x.shape}")
    print(f"输入时间步形状: {dummy_t.shape}")
    print(f"输入类别标签形状: {dummy_c.shape}")
    print(f"输出 (预测噪声) 形状: {predicted_noise.shape}")

    assert dummy_x.shape == predicted_noise.shape, "输入和输出的形状不匹配!"

    print("\n模型测试通过！输入输出形状一致。")
