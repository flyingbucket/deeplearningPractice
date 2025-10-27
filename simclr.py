# simclr_cpu_demo.py
import math
import random
from tqdm import tqdm
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# ========== 1. 随机种子，确保可复现 ==========
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


set_seed(42)
device = torch.device("cpu")


# ========== 2. 数据增强：两视角 ==========
class TwoCropsTransform:
    """对同一图像做两次随机增强，得到正样本对"""

    def __init__(self):
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=32, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
            ]
        )

    def __call__(self, x):
        return self.transform(x), self.transform(x)


# 评估时仅做最小增强
test_transform = transforms.Compose([transforms.ToTensor()])


# ========== 3. 数据集（默认下采样一部分以提速） ==========
def get_cifar10(limit_train=10000, limit_test=2000, root="./data"):
    train_full = datasets.CIFAR10(
        root=root, train=True, download=True, transform=TwoCropsTransform()
    )
    test_full = datasets.CIFAR10(
        root=root, train=False, download=True, transform=test_transform
    )

    if limit_train is not None and limit_train < len(train_full):
        train_idx = list(range(limit_train))
        train = Subset(train_full, train_idx)
    else:
        train = train_full

    if limit_test is not None and limit_test < len(test_full):
        test_idx = list(range(limit_test))
        test = Subset(test_full, test_idx)
    else:
        test = test_full

    return train, test


# ========== 4. 小型 CNN 编码器 + 投影头 ==========
class SmallCNN(nn.Module):
    def __init__(self, feat_dim=256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # 全局池化 -> (B,256,1,1)
        )
        self.fc = nn.Linear(256, feat_dim)

    def forward(self, x):
        x = self.features(x).flatten(1)  # (B,256)
        x = self.fc(x)  # (B,feat_dim)
        return x


class ProjectionHead(nn.Module):
    def __init__(self, in_dim=256, out_dim=128, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        return self.net(x)


# ========== 5. InfoNCE（NT-Xent）对比损失 ==========
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.t = temperature

    def forward(self, z1, z2):
        """
        z1, z2: (N, D) 两个视角的特征
        先做L2归一化 -> 拼接成 (2N, D)
        相似度矩阵 sim = z @ z.T (余弦)
        对每个样本，其正样本是另一个视角对应的同索引样本
        """
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        z = torch.cat([z1, z2], dim=0)  # (2N, D)
        sim = torch.mm(z, z.t()) / self.t  # (2N, 2N)

        N = z1.size(0)
        labels = torch.arange(2 * N)
        labels = labels.roll(shifts=N)  # i 的正样本是 i^N（跨视角）
        # 屏蔽对角（自身），否则会把自己当成正样本
        mask = torch.eye(2 * N, dtype=torch.bool)
        sim = sim.masked_fill(mask, -1e9)

        loss = F.cross_entropy(sim, labels.to(sim.device))
        return loss


# ========== 6. 训练 SimCLR 表征 ==========
def train_simclr(
    epochs=10,
    batch_size=128,
    lr=1e-3,
    temperature=0.5,
    limit_train=10000,
    limit_test=2000,
):
    train_set, test_set = get_cifar10(limit_train=limit_train, limit_test=limit_test)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2
    )

    encoder = SmallCNN(feat_dim=256).to(device)
    projector = ProjectionHead(in_dim=256, out_dim=128).to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(projector.parameters()), lr=lr
    )
    criterion = NTXentLoss(temperature=temperature)

    encoder.train()
    projector.train()
    for ep in range(1, epochs + 1):
        total_loss = 0.0
        for (x1, x2), _ in tqdm(train_loader, desc=f"epoch:{ep}/{epochs}"):
            x1, x2 = x1.to(device), x2.to(device)
            h1 = encoder(x1)
            h2 = encoder(x2)
            z1 = projector(h1)
            z2 = projector(h2)
            loss = criterion(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        tqdm.write(f"[SimCLR] Epoch {ep}/{epochs}  loss={avg_loss:.4f}")

    return encoder, test_loader


# ========== 7. 线性探针（冻结编码器） ==========
class LinearProbe(nn.Module):
    def __init__(self, in_dim=256, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


@torch.no_grad()
def extract_features(dataloader, encoder):
    encoder.eval()
    feats, labels = [], []
    for batch in dataloader:
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            x, y = batch
        else:
            x, y = batch[0], batch[1]
        x = x.to(device)
        if isinstance(x, (list, tuple)):  # 训练集是 TwoCropsTransform，测试集不是
            x = x[0]  # 取一视角即可
        h = encoder(x)
        feats.append(h.cpu())
        labels.append(y)
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


def train_linear_probe(
    encoder, train_limit=10000, test_limit=2000, epochs=5, batch_size=256, lr=1e-2
):
    # 重新加载无增强的数据集作为线性评估用
    train_set_full = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=test_transform
    )
    test_set_full = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform
    )
    if train_limit is not None and train_limit < len(train_set_full):
        train_set = Subset(train_set_full, list(range(train_limit)))
    else:
        train_set = train_set_full
    if test_limit is not None and test_limit < len(test_set_full):
        test_set = Subset(test_set_full, list(range(test_limit)))
    else:
        test_set = test_set_full

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # 提取特征
    train_feats, train_labels = extract_features(train_loader, encoder)
    test_feats, test_labels = extract_features(test_loader, encoder)

    probe = LinearProbe(in_dim=train_feats.size(1), num_classes=10).to(device)
    opt = torch.optim.SGD(probe.parameters(), lr=lr, momentum=0.9)
    ce = nn.CrossEntropyLoss()

    # 把特征当作常量张量来训练线性分类器
    N = train_feats.size(0)
    idx = torch.arange(N)

    for ep in range(1, epochs + 1):
        # 简单的小批量循环
        perm = torch.randperm(N)
        total_loss = 0.0
        probe.train()

        pbar = tqdm(range(0, N, batch_size), desc=f"Epoch {ep}/{epochs}")
        for i in pbar:
            batch_idx = perm[i : i + batch_size]
            x = train_feats[batch_idx].to(device)
            y = train_labels[batch_idx].to(device)
            logits = probe(x)
            loss = ce(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        avg_loss = total_loss / math.ceil(N / batch_size)

        # 验证
        probe.eval()
        with torch.no_grad():
            logits = probe(test_feats.to(device))
            pred = logits.argmax(dim=1).cpu()
            acc = (pred == test_labels).float().mean().item()
        print(
            f"[LinearProbe] Epoch {ep}/{epochs}  loss={avg_loss:.4f}  test_acc={acc * 100:.2f}%"
        )


def main():
    encoder, _ = train_simclr(
        epochs=10,  # 可适当增减
        batch_size=128,  # CPU 建议 64~128
        lr=1e-3,
        temperature=0.5,
        limit_train=10000,  # 用 1/5 训练集加速
        limit_test=2000,
    )
    train_linear_probe(
        encoder, train_limit=10000, test_limit=2000, epochs=5, batch_size=256, lr=1e-2
    )


if __name__ == "__main__":
    main()
