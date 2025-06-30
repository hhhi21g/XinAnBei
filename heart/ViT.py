import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch.nn.functional as F
from timm import create_model

# ✅ FocalLoss 实现
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
        return loss.mean() if self.reduction == "mean" else loss.sum()

# ✅ 路径配置
train_dir = "output/bispec_output/train"
val_dir = "output/bispec_output/val"
test_dir = "output/bispec_output/test"

# ✅ 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ ViT 模型加载
model = create_model("vit_base_patch16_224", pretrained=True, num_classes=1)  # 输出1个节点用于 sigmoid
model = model.to(device)

# ✅ 图像增强和预处理
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)  # 可换为 ImageNet 均值
])
transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ✅ 数据加载
train_set = datasets.ImageFolder(train_dir, transform=transform_train)
val_set = datasets.ImageFolder(val_dir, transform=transform_val)
test_set = datasets.ImageFolder(test_dir, transform=transform_val)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32)
test_loader = DataLoader(test_set, batch_size=32)

# ✅ 损失函数与优化器
criterion = BinaryFocalLoss(gamma=2.0, alpha=0.8, reduction="mean").to(device)
optimizer = optim.AdamW(model.parameters(), lr=3e-5)

# ✅ 训练
best_val_loss = float("inf")
num_epochs = 20

# for epoch in range(num_epochs):
#     model.train()
#     total_loss = 0
#     loop = tqdm(train_loader, desc=f"📘 Epoch {epoch+1}/{num_epochs}")
#     for inputs, labels in loop:
#         inputs, labels = inputs.to(device), labels.to(device).float()
#         optimizer.zero_grad()
#         outputs = model(inputs).squeeze(1)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#         loop.set_postfix(loss=loss.item())
#
#     print(f"✅ Epoch {epoch+1} completed. Train Loss: {total_loss:.4f}")
#
#     # ✅ 验证
#     # ✅ 验证阶段：寻找最佳阈值
#     model.eval()
#     val_loss = 0.0
#     all_probs, all_targets = [], []
#
#     with torch.no_grad():
#         for inputs, targets in val_loader:
#             inputs = inputs.to(device)
#             targets = targets.to(device).float()
#             outputs = model(inputs).squeeze(1)
#             loss = criterion(outputs, targets)
#             val_loss += loss.item()
#
#             probs = torch.sigmoid(outputs).detach().cpu().numpy()
#             all_probs.extend(probs)
#             all_targets.extend(targets.cpu().numpy())
#
#     avg_val_loss = val_loss / len(val_loader)
#     print(f"🔍 验证损失: {avg_val_loss:.4f}")
#
#     # ✅ 阈值搜索
#     from sklearn.metrics import f1_score
#     import numpy as np
#
#     best_threshold = 0.5
#     best_f1 = 0.0
#     for threshold in np.arange(0.1, 0.9, 0.01):
#         preds = (np.array(all_probs) > threshold).astype(int)
#         f1 = f1_score(all_targets, preds)
#         if f1 > best_f1:
#             best_f1 = f1
#             best_threshold = threshold
#
#     print(f"🎯 最佳阈值: {best_threshold:.2f}，对应验证F1分数: {best_f1:.4f}")
#
#     if avg_val_loss < best_val_loss:
#         best_val_loss = avg_val_loss
#         torch.save({
#             'model': model.state_dict(),
#             'threshold': best_threshold
#         }, "best_vit_model.pth")
#         print("✅ 保存当前最优模型和阈值（验证损失最低）")

# ✅ 加载模型和最佳阈值
checkpoint = torch.load("three_models\\best_vit_model.pth")
model.load_state_dict(checkpoint['model'])
best_threshold = checkpoint.get('threshold', 0.5)

# ✅ 测试
model.eval()
test_preds, test_labels = [], []
with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="🧪 Testing"):
        inputs = inputs.to(device)
        labels = labels.to(device).float()
        outputs = model(inputs).squeeze(1)
        probs = torch.sigmoid(outputs)
        preds = (probs > best_threshold).long()
        test_preds.extend(preds.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())
