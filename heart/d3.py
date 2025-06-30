import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from timm import create_model
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms


# ✅ 路径配置
DATA_DIR = "/kaggle/working/bispec_output"  # 你的 bispectrum 图像目录
BATCH_SIZE = 64
EPOCHS = 8
NUM_CLASSES = 2
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ✅ Mixup 函数
def mixup_data(x, y, alpha=0.4):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ✅ CLANE 模拟类
class CLANE(object):
    def __init__(self, mask_ratio=0.2):
        self.mask_ratio = mask_ratio

    def __call__(self, img):
        img_np = np.array(img).copy()
        h, w, c = img_np.shape
        mask_h = int(h * self.mask_ratio)
        mask_w = int(w * self.mask_ratio)
        top = random.randint(0, h - mask_h)
        left = random.randint(0, w - mask_w)
        img_np[top:top + mask_h, left:left + mask_w, :] = 0
        return Image.fromarray(img_np)


class RandomCLANE(object):
    def __init__(self, p=0.5, mask_ratio=0.2):
        self.p = p
        self.clane = CLANE(mask_ratio)

    def __call__(self, img):
        if random.random() < self.p:
            return self.clane(img)
        return img


# ✅ 图像增强
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomRotation(degrees=3),
    # transforms.RandomVerticalFlip(p=0.3),
    # transforms.RandomRotation(degrees=2),
    RandomCLANE(p=0.4, mask_ratio=0.25),  # ⬅️ CLANE 添加
    # AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.1)
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ✅ 数据加载（确保每个 split 文件夹下有 0/1 子文件夹）
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_transform)
test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ✅ 模型构建
model = create_model("convnext_tiny", pretrained=True, num_classes=NUM_CLASSES)
model.to(DEVICE)

# ✅ 损失函数 + 优化器
criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.8, 1.2]).to(DEVICE))
optimizer = torch.optim.AdamW(model.parameters(), lr=3.5e-4, weight_decay=0.01)

total_steps = len(train_loader) * EPOCHS
warmup_steps = int(0.2 * total_steps)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)


# ✅ 训练函数
def train_epoch(model, loader):
    model.train()
    total_loss, correct = 0, 0
    for imgs, labels in tqdm(loader, desc="🟢 Train"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        # imgs, y_a, y_b, lam = mixup_data(imgs, labels)
        optimizer.zero_grad()
        outputs = model(imgs)
        # loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * imgs.size(0)
        pred = outputs.argmax(1)
        correct += (pred == labels).sum().item()
        # correct += (lam * pred.eq(y_a).sum().item() + (1 - lam) * pred.eq(y_b).sum().item())
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


# ✅ 验证函数
def eval_model(model, loader, split="Val"):
    model.eval()
    correct, preds, trues = 0, [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc=f"🔵 {split}"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            pred_labels = outputs.argmax(1)
            correct += (pred_labels == labels).sum().item()
            preds.extend(pred_labels.cpu().numpy())
            trues.extend(labels.cpu().numpy())
    acc = correct / len(loader.dataset)
    return acc, preds, trues


# ✅ 主训练循环
best_val_acc = 0
for epoch in range(EPOCHS):
    print(f"\n📘 Epoch {epoch + 1}/{EPOCHS}")
    train_loss, train_acc = train_epoch(model, train_loader)
    val_acc, _, _ = eval_model(model, val_loader)
    print(f"✅ Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_convnext_tiny.pth")

# ✅ 最终测试评估
# ✅ 最终测试评估
model.load_state_dict(torch.load("best_convnext_tiny.pth"))
test_acc, y_pred, y_true = eval_model(model, test_loader, split="Test")
print(f"\n✅ 测试集准确率: {test_acc:.4f}")
print("📊 分类报告:\n", classification_report(y_true, y_pred))

import seaborn as sns
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, cmap="Greens", fmt="d",
            xticklabels=["Abnormal", "Normal"], yticklabels=["Abnormal", "Normal"])
plt.title("Test Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

