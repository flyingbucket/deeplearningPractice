import os
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np
from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

NUM_CLASSES = 21
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
voc_dir = "./data/VOCdevkit/VOC2012"
VOC_COLORMAP = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
]

VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "potted plant",
    "sheep",
    "sofa",
    "train",
    "tv/monitor",
]


def read_voc_images(voc_dir, is_train=True):
    """读取所有VOC图像并标注"""
    txt_fname = os.path.join(
        voc_dir, "ImageSets", "Segmentation", "train.txt" if is_train else "val.txt"
    )
    mode = torchvision.io.image.ImageReadMode.RGB
    with open(txt_fname, "r") as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        features.append(
            torchvision.io.read_image(
                os.path.join(voc_dir, "JPEGImages", f"{fname}.jpg")
            )
        )
        labels.append(
            torchvision.io.read_image(
                os.path.join(voc_dir, "SegmentationClass", f"{fname}.png"), mode
            )
        )
    return features, labels


def voc_colormap2label():
    colormap2label = torch.full((256**3,), 255, dtype=torch.long)  # default 255=ignore
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label


def voc_label_indices(colormap, colormap2label):
    colormap = colormap.permute(1, 2, 0).numpy().astype("int32")  # H,W,3
    idx = (colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256 + colormap[:, :, 2]
    idx_t = torch.from_numpy(idx).long()
    return colormap2label[idx_t]  # (H,W) long, 0..20 或 255(忽略)


def voc_rand_crop(feature, label, height, width):
    """随机裁剪特征和标签图像"""
    rect = torchvision.transforms.RandomCrop.get_params(feature, (height, width))
    feature = torchvision.transforms.functional.crop(feature, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label


class VOCSegDataset(torch.utils.data.Dataset):
    """一个用于加载VOC数据集的自定义数据集"""

    def __init__(self, is_train, crop_size, voc_dir):
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        self.features = [
            self.normalize_image(feature) for feature in self.filter(features)
        ]
        self.labels = self.filter(labels)
        self.colormap2label = voc_colormap2label()
        print("read " + str(len(self.features)) + " examples")

    def normalize_image(self, img):
        return self.transform(img.float() / 255)

    def filter(self, imgs):
        return [
            img
            for img in imgs
            if (img.shape[1] >= self.crop_size[0] and img.shape[2] >= self.crop_size[1])
        ]

    def __getitem__(self, idx):
        feature, label = voc_rand_crop(
            self.features[idx], self.labels[idx], *self.crop_size
        )
        return (feature, voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)


def build_model(num_classes=NUM_CLASSES):
    """
    迁移学习套路：
    1) 载入在 COCO 上预训练的 DeepLabV3-ResNet50
    2) 替换分类头为 21 类（VOC）
    """
    model = models.segmentation.deeplabv3_resnet50(
        weights=models.segmentation.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
    )
    # 替换主分类头
    model.classifier = DeepLabHead(2048, num_classes)
    # 可选：移除/关闭 aux_classifier（保持简洁）
    model.aux_classifier = None
    return model


# --------- 指标：mIoU / Pixel Acc ----------
def _fast_hist(true, pred, num_classes):
    mask = (true >= 0) & (true < num_classes)
    hist = np.bincount(
        num_classes * true[mask].astype(int) + pred[mask], minlength=num_classes**2
    ).reshape(num_classes, num_classes)
    return hist


@torch.no_grad()
def evaluate(model, dataloader, device=DEVICE, num_classes=NUM_CLASSES):
    model.eval()
    hist = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_correct, total_label = 0, 0

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)  # (N,H,W) long with 0..20 or 255

        outputs = model(images)["out"]  # (N,C,H,W)
        preds = outputs.argmax(1)  # (N,H,W)

        valid = targets != 255
        total_correct += (preds[valid] == targets[valid]).sum().item()
        total_label += valid.sum().item()

        # 统计直方图（转 numpy）
        for lt, lp in zip(targets.cpu().numpy(), preds.cpu().numpy()):
            # 跳过 ignore 像素
            m = lt != 255
            hist += _fast_hist(lt[m], lp[m], num_classes)

    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-10)
    miou = np.nanmean(iu)
    pix_acc = total_correct / (total_label + 1e-10)
    return miou, pix_acc


# --------- 训练一个 epoch ----------
def train_one_epoch(
    model, dataloader, criterion, optimizer, scaler, device=DEVICE, max_norm=0.0
):
    model.train()
    running_loss = 0.0
    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=torch.cuda.is_available()):
            out = model(images)["out"]  # (N,C,H,W)
            loss = criterion(out, targets)  # CE with ignore_index

        scaler.scale(loss).backward()
        if max_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(dataloader.dataset)


# --------- 主入口（超参数可按需调整） ----------
def main_train(
    batch_size=8,
    num_workers=12,
    crop_size=(320, 480),
    lr=1e-3,
    weight_decay=1e-4,
    epochs=20,
    max_norm=0.0,
):
    # DataLoader（你已有的 Dataset 保证了固定尺寸裁剪，默认 collate 就能 batch）
    train_ds = VOCSegDataset(True, crop_size, voc_dir)
    val_ds = VOCSegDataset(False, crop_size, voc_dir)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Model / Loss / Optim / Scheduler
    model = build_model(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=255)  # 关键：忽略 void 像素
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    best_miou = 0.0
    for ep in tqdm(range(1, epochs + 1), desc="training"):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, DEVICE, max_norm
        )
        miou, pixacc = evaluate(model, val_loader, DEVICE, NUM_CLASSES)
        scheduler.step()

        print(
            f"[Epoch {ep:02d}] loss={train_loss:.4f}  mIoU={miou:.4f}  PA={pixacc:.4f}  lr={scheduler.get_last_lr()[0]:.2e}"
        )

        # 保存最好模型
        if miou > best_miou:
            best_miou = miou
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(
                {"epoch": ep, "state_dict": model.state_dict()},
                f"checkpoints/best_deeplabv3_resnet50_voc.pth",
            )
            print(f"  ↳ New best mIoU: {best_miou:.4f} (model saved)")


if __name__ == "__main__":
    main_train(
        batch_size=8,
        num_workers=12,
        crop_size=(320, 480),
        lr=1e-4,  # 迁移学习建议从 1e-4~5e-4 起步
        weight_decay=1e-4,
        epochs=30,
        max_norm=0.0,  # 如需梯度裁剪可设 1.0
    )
