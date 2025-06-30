# wav2vec2训练
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Processor, Wav2Vec2Model, get_cosine_schedule_with_warmup
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve
from torch import nn, optim
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 参数配置
AUG_DATA_DIR = "/kaggle/working/metadata"  # 保存增强数据的路径
MAX_LEN = 16000 * 10
BATCH_SIZE = 16
EPOCHS = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载处理器,从hugging face
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")


# 数据集
class NPYDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        waveform = np.load(row["npy_path"])
        return waveform, row["target"]


# Batch 组装
def collate_fn(batch):
    waveforms, labels = zip(*batch)
    inputs = processor(list(waveforms), sampling_rate=16000, return_tensors="pt", padding=True, truncation=True,
                       max_length=MAX_LEN)
    labels = torch.tensor(labels, dtype=torch.float32)
    return inputs.input_values, labels


# 模型结构
class Wav2Vec2Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, input_values):
        out = self.wav2vec2(input_values).last_hidden_state
        pooled = out.mean(dim=1)
        return self.classifier(pooled).squeeze(1)


# 验证评估
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)


# 加载数据
train_loader = DataLoader(NPYDataset(os.path.join(AUG_DATA_DIR, "train_meta.csv")), batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=collate_fn)
val_loader = DataLoader(NPYDataset(os.path.join(AUG_DATA_DIR, "val_meta.csv")), batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=collate_fn)
test_loader = DataLoader(NPYDataset(os.path.join(AUG_DATA_DIR, "test_meta.csv")), batch_size=BATCH_SIZE, shuffle=False,
                         collate_fn=collate_fn)

# 初始化模型与优化器
model = Wav2Vec2Classifier().to(device)
for name, param in model.wav2vec2.named_parameters():
    param.requires_grad = True

# 统计样本数量
train_df = pd.read_csv(os.path.join(AUG_DATA_DIR, "train_meta.csv"))
num_pos = train_df["target"].sum()
num_neg = len(train_df) - num_pos
pos_weight = torch.tensor([num_neg / num_pos], device=device)  # 注意加 []

criterion = nn.BCELoss()

optimizer = optim.AdamW([
    {"params": [p for n, p in model.wav2vec2.named_parameters() if p.requires_grad], "lr": 1e-5},
    {"params": model.classifier.parameters(), "lr": 3.5e-4}
])

# 学习率调度器
total_steps = len(train_loader) * EPOCHS
warmup_steps = int(total_steps * 0.2)
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

# 训练主循环
best_val_loss = float('inf')
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    avg_val_loss = evaluate(model, val_loader, criterion)
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_model_w.pth")
        print(f"最优模型已保存 (第{epoch + 1}轮)")
    print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

# 阈值选择
val_preds, val_targets = [], []
model.eval()
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        outputs = model(inputs).cpu().numpy()
        val_preds.extend(outputs)
        val_targets.extend(labels.numpy())

precision, recall, thresholds = precision_recall_curve(val_targets, val_preds)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"验证集最佳阈值: {best_threshold:.4f}")

model.load_state_dict(torch.load("best_model_wav2vec2.pth", map_location=device))
model.eval()

# 测试集推理与评估
test_preds, test_labels = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs).cpu().numpy()
        test_preds.extend(outputs)
        test_labels.extend(labels.numpy())

binary_preds = [1 if p > best_threshold else 0 for p in test_preds]
acc = accuracy_score(test_labels, binary_preds)
auc = roc_auc_score(test_labels, test_preds)
print(f"测试集 Accuracy: {acc:.4f}, AUC: {auc:.4f}")

# 混淆矩阵计算
cm = confusion_matrix(test_labels, binary_preds)
print("混淆矩阵：")
print(cm)

# 可视化
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.grid(False)
plt.show()
