from torchvision.models import resnet18
import torch.nn as nn
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建模型并修改输出层
model = resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)  # ➤ 若是 sigmoid 二分类
checkpoint = torch.load("MedViTV2/resnet18_'PAD'.pth", map_location=device)
model.load_state_dict(checkpoint['model'])  # 🔧 注意是 checkpoint['model']

model.to(device)

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_dir = "output\\bispec_output\\test"
test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32)

all_preds, all_labels = [], []

with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="🧪 Testing"):
        inputs = inputs.to(device)
        labels = labels.to(device).float()
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ✅ 分析结果
print(f"\n🎯 测试准确率: {accuracy_score(all_labels, all_preds):.4f}")
print("📊 分类报告:\n", classification_report(all_labels, all_preds, digits=4))

# ✅ 混淆矩阵
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=["Normal", "Abnormal"],
            yticklabels=["Normal", "Abnormal"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()