import os
from torchvision import datasets

# 目标目录（Windows 路径用原始字符串 r"" 避免反斜杠转义）
ROOT = r""
os.makedirs(ROOT, exist_ok=True)

# 下载训练集与测试集；会在 ROOT 下创建 MNIST/raw 和 MNIST/processed
train_ds = datasets.MNIST(root=ROOT, train=True,  download=True)
test_ds  = datasets.MNIST(root=ROOT, train=False, download=True)

mnist_dir = os.path.join(ROOT, "MNIST")
print("MNIST 已保存到：", mnist_dir)
print("训练集大小：", len(train_ds), "测试集大小：", len(test_ds))

# 可选：打印一下关键文件，确认下载成功
for sub in ("raw", "processed"):
    p = os.path.join(mnist_dir, sub)
    if os.path.isdir(p):
        print(sub, "包含：", os.listdir(p))

