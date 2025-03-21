---
title: 用tensorboard支持pytorch训练可视化
mathjax: true
toc: true
date: 2025-03-22 03:00:23
updated: 2025-03-22 03:00:23
categories: 
- PyTorch
tags:
- TensorBoard
- Visualization
---
在工作用了tensorboard来可视化模型训练过程后，发现还挺香的。另外pytorch也正式支持tensorboard了，这里记录一下。

<!--more-->

## 前置条件

安装tensorboard：

```zsh
pip install tensorboard
```

## 实现步骤

1. 指定tensorboard输出日志：`writer = SummaryWriter(log_dir=LOG_DIR)`
2. 将模型和数据集添加到writer中：`writer.add_graph(model, images.to(device))`
3. 记录过程数据指标：`writer.add_scalar('Test Loss', avg_loss, epoch)`
4. 当模型开始训练后，启动tensorboard：`tensorboard --logdir=runs`。打开链接就能看到模型过程指标了：[http://localhost:6006/](http://localhost:6006/)

## 代码示例

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# 1. 设置参数
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
NUM_CLASSES = 10
LOG_DIR = "runs/fashion_mnist_experiment_" + datetime.now().strftime("%Y%m%d_%H%M%S")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 准备数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_set = torchvision.datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_set = torchvision.datasets.FashionMNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# 3. 定义模型
class FashionMNISTModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = FashionMNISTModel(NUM_CLASSES).to(device)

# 4. 初始化TensorBoard Writer
writer = SummaryWriter(log_dir=LOG_DIR)

# 5. 添加模型结构和数据集到TensorBoard
images, _ = next(iter(train_loader))
# note: 模型和数据集要么都在cpu，要么都在gpu；不然报错
writer.add_graph(model, images.to(device))

# 6. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# 7. 训练循环
def train():
    model.train()
    # 用累加loss，不然单个batch loss下降不明显
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # 每100个batch记录一次
        if batch_idx % 100 == 0:
            writer.add_scalar('Training Loss',
                              loss.item(),
                              epoch * len(train_loader) + batch_idx)
            running_loss = 0


# 8. 测试函数
def test():
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = test_loss / len(test_loader)

    # 记录测试结果
    writer.add_scalar('Test Loss', avg_loss, epoch)
    writer.add_scalar('Test Accuracy', accuracy, epoch)

    print(f"Epoch [{epoch + 1}/{EPOCHS}], "
          f"Test Loss: {avg_loss:.4f}, "
          f"Test Accuracy: {accuracy:.2f}%")


# 9. 主训练循环
for epoch in range(EPOCHS):
    train()
    test()

# 10. 关闭Writer
writer.close()

print("训练完成！")
```


