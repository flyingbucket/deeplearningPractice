import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

import random
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class CNNClassifier(nn.Module):
    """
    基于CNN的图片分类器
    """

    def __init__(self, num_features, num_classes, num_convs=3) -> None:
        super().__init__()

        layers = []
        in_channels = 3
        for _ in range(num_convs):
            layers.append(
                nn.Conv2d(
                    in_channels,
                    num_features,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(num_features))
            layers.append(nn.ReLU(inplace=True))
            in_channels = num_features
        self.conv_norm = nn.Sequential(*layers)
        self.pooling = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.conv_norm(x)
        x = self.pooling(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def generate_colortemp_dummy(batch_size=32, img_size=32, jitter=30):
    """
    生成 dummy RGB 图像 (纯色块)，根据色温分为 5 类，并添加随机抖动。

    类别:
      0: 冷蓝
      1: 蓝绿
      2: 中性白
      3: 暖黄
      4: 暖红
    """
    # 基础颜色 (中心色)
    colortemp_classes = {
        0: (0, 102, 255),  # 冷蓝
        1: (0, 255, 255),  # 蓝绿
        2: (255, 255, 255),  # 中性白
        3: (255, 255, 102),  # 暖黄
        4: (255, 102, 0),  # 暖红
    }

    images = []
    labels = []
    for _ in range(batch_size):
        label = np.random.randint(0, 5)  # 随机类别
        base_color = np.array(colortemp_classes[label], dtype=np.int16)

        # 随机扰动
        noise = np.random.randint(-jitter, jitter + 1, size=3)
        color = np.clip(base_color + noise, 0, 255).astype(np.uint8)

        # 生成整张图片
        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * color
        images.append(img)
        labels.append(label)

    images = np.stack(images)  # (B, H, W, 3)
    labels = np.array(labels)

    images_tensor = torch.from_numpy(images).float().permute(0, 3, 1, 2) / 255.0
    labels_tensor = torch.from_numpy(labels).long()

    return images, images_tensor, labels_tensor


if __name__ == "__main__":
    # ===== 1) 可复现性 & 设备 =====
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # dummy dataset
    N_train = 4000
    N_val = 1000
    img_size = 32

    _, X_train, y_train = generate_colortemp_dummy(
        batch_size=N_train, img_size=img_size, jitter=40
    )
    _, X_val, y_val = generate_colortemp_dummy(
        batch_size=N_val, img_size=img_size, jitter=40
    )

    # DataLoader
    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=128, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), batch_size=256, shuffle=False, num_workers=0
    )

    # model and loss
    num_features = 32
    num_classes = 5
    model = CNNClassifier(
        num_features=num_features, num_classes=num_classes, num_convs=3
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # training
    epochs = 20
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += xb.size(0)
            pbar.set_postfix(loss=loss.item(), acc=f"{(correct / total) * 100:.2f}%")

        train_loss = running_loss / total
        train_acc = correct / total

        # test model
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss_sum = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss_sum += loss.item() * xb.size(0)
                pred = logits.argmax(dim=1)
                val_correct += (pred == yb).sum().item()
                val_total += xb.size(0)

        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total
        print(
            f"[Epoch {epoch}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc

    print(f"Best Val Acc: {best_val_acc * 100:.2f}%")

    # test and inference
    # 再生成一些全新的样本做推理演示
    _, X_test, y_test = generate_colortemp_dummy(
        batch_size=20, img_size=img_size, jitter=40
    )
    X_test = X_test.to(device)
    with torch.no_grad():
        logits = model(X_test)
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1).cpu()

    acc = (pred == y_test).float().mean().item()
    print(f"Toy Test Acc on 20 samples: {acc * 100:.2f}%")

    # 可视化前 10 张预测
    fig, axes = plt.subplots(2, 10, figsize=(18, 4))
    axes = axes.flatten()
    imgs_uint8, _, _ = generate_colortemp_dummy(
        batch_size=20, img_size=img_size, jitter=40
    )
    for i in range(20):
        axes[i].imshow(imgs_uint8[i])
        axes[i].set_title(f"pred:{pred[i].item()} | gt:{y_test[i].item()}")
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()
