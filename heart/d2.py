# efficientNet训练
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.transforms import AutoAugmentPolicy, AutoAugment
from tqdm import tqdm
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_factor = (1 - p_t).pow(self.gamma)
        loss = self.alpha * focal_factor * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# === 路径设置 ===
train_dir = "output\\bispec_output\\train"
val_dir = "output\\bispec_output\\val"
test_dir = "output\\bispec_output\\test"

# === 模型加载 ===
weights = EfficientNet_B1_Weights.DEFAULT
model = efficientnet_b1(weights=weights)

# ⚠️ 修改输出层为单节点，表示有病概率（适合 sigmoid）
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# ✅ Mixup 函数
# def mixup_data(x, y, alpha=0.4):
#     if alpha > 0:
#         lam = np.random.beta(alpha, alpha)
#     else:
#         lam = 1
#     batch_size = x.size(0)
#     index = torch.randperm(batch_size).to(x.device)
#     mixed_x = lam * x + (1 - lam) * x[index, :]
#     y_a, y_b = y, y[index]
#     return mixed_x, y_a, y_b, lam
#
#
# def mixup_criterion(criterion, pred, y_a, y_b, lam):
#     return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
#
#
# # ✅ CLANE 模拟类
# class CLANE(object):
#     def __init__(self, mask_ratio=0.2):
#         self.mask_ratio = mask_ratio
#
#     def __call__(self, img):
#         img_np = np.array(img).copy()
#         h, w, c = img_np.shape
#         mask_h = int(h * self.mask_ratio)
#         mask_w = int(w * self.mask_ratio)
#         top = random.randint(0, h - mask_h)
#         left = random.randint(0, w - mask_w)
#         img_np[top:top + mask_h, left:left + mask_w, :] = 0
#         return Image.fromarray(img_np)
#
#
# class RandomCLANE(object):
#     def __init__(self, p=0.5, mask_ratio=0.2):
#         self.p = p
#         self.clane = CLANE(mask_ratio)
#
#     def __call__(self, img):
#         if random.random() < self.p:
#             return self.clane(img)
#         return img


# ✅ 图像增强
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomRotation(degrees=2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]) # 可换为 ImageNet 均值
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === 加载数据集 ===
train_set = datasets.ImageFolder(train_dir, transform=train_transform)
val_set = datasets.ImageFolder(val_dir, transform=transform_val)
test_set = datasets.ImageFolder(test_dir, transform=transform_val)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32)
test_loader = DataLoader(test_set, batch_size=32)

# === 损失函数 & 优化器 ===
criterion = BinaryFocalLoss(gamma=2.0, alpha=0.8, reduction="mean").to(device)
optimizer = optim.AdamW(model.parameters(), lr=3.5e-4, weight_decay=0.01)

# === 训练 ===
best_val_loss = float("inf")
num_epochs = 10
#
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, desc=f"📘 Epoch {epoch + 1}/{num_epochs}")
    for inputs, labels in loop:
        inputs, labels = inputs.to(device), labels.to(device).float()
        optimizer.zero_grad()
        outputs = model(inputs).squeeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    print(f"✅ Epoch {epoch + 1} completed. Train Loss: {total_loss:.4f}")

    # === 验证阶段：计算验证损失 ===
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).float()
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"🔍 验证损失: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_model.pth")
        print("✅ 保存当前最优模型（验证损失最低）")

# === 测试 ===
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
test_preds, test_labels = [], []
with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="🧪 Testing"):
        inputs = inputs.to(device)
        labels = labels.to(device).float()
        outputs = model(inputs).squeeze(1)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.45).long()
        test_preds.extend(preds.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

acc = accuracy_score(test_labels, test_preds)
print(f"\n✅ 测试集准确率: {acc:.4f}")
print("📊 分类报告:\n", classification_report(test_labels, test_preds))
sns.heatmap(confusion_matrix(test_labels, test_preds), annot=True, cmap="Greens", fmt="d",
            xticklabels=["Abormal", "Normal"], yticklabels=["Abnormal", "Normal"])
plt.title("Test Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()
