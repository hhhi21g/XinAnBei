import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torchvision import transforms
from torchvision.models import efficientnet_b0, efficientnet_b1
from PIL import Image
from tqdm import tqdm
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import timm
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

efficientnet = efficientnet_b1(weights=None)  # ⚠️ 一定是 b1，不是 b0
efficientnet.classifier[1] = nn.Linear(efficientnet.classifier[1].in_features, 1)
# 加载权重
efficientnet.load_state_dict(torch.load("three_models\\best_model.pth", map_location=DEVICE))
efficientnet.to(DEVICE).eval()

# ✅ ConvNeXt-Tiny 加载（必须是 num_classes=1）
convnext = timm.create_model("convnext_tiny", pretrained=False, num_classes=2)
convnext.load_state_dict(torch.load("three_models\\best_convnext_tiny.pth", map_location=DEVICE))
convnext = convnext.to(DEVICE).eval()

# ✅ Wav2Vec2 模型定义
class Wav2Vec2Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("wav2vec2-base")
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
        with torch.no_grad():
            out = self.wav2vec2(input_values).last_hidden_state
        pooled = out.mean(dim=1)
        return self.classifier(pooled).squeeze(1)

# ✅ 加载 Wav2Vec2 模型
wav_model = Wav2Vec2Classifier().to(DEVICE)
wav_model.load_state_dict(torch.load("three_models\\best_model_w.pth", map_location=DEVICE))
wav_model.eval()

# ✅ 图像处理
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ✅ 处理器
processor = Wav2Vec2Processor.from_pretrained("wav2vec2-base")

# ✅ 预测函数
def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        p1 = torch.sigmoid(efficientnet(img)).squeeze()           # shape: []
        p2_logits = convnext(img)
        p2 = F.softmax(p2_logits, dim=1)[:, 1].squeeze()           # shape: []
    return p1.item(), p2.item()

def predict_audio(wav_path):
    if wav_path.endswith(".npy"):
        waveform = np.load(wav_path)
        waveform = torch.tensor(waveform, dtype=torch.float32)
    else:
        waveform, _ = torchaudio.load(wav_path)
        waveform = waveform.squeeze(0)
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(DEVICE)
    with torch.no_grad():
        p3 = wav_model(input_values)
    return p3.item()

# ✅ 加载 test_meta.csv
meta_csv = "output\\metadata\\test_meta.csv"
df = pd.read_csv(meta_csv)

results = []

print("🔍 开始融合推理...")
for _, row in tqdm(df.iterrows(), total=len(df)):
    img_path = row["img_path"]
    wav_path = row["npy_path"]
    label = row["target"]

    if not os.path.exists(img_path) or not os.path.exists(wav_path):
        print(f"⚠️ 缺失文件: {img_path} 或 {wav_path}")
        continue

    try:
        p1, p2 = predict_image(img_path)
        p3 = predict_audio(wav_path)
        avg_prob = np.mean([p1, p2, p3])
        pred = int(avg_prob >= 0.5)

        results.append({
            "img_path": img_path,
            "npy_path": wav_path,
            "true_label": label,
            "p1": round(p1, 4),
            "p2": round(p2, 4),
            "p3": round(p3, 4),
            "avg_prob": round(avg_prob, 4),
            "fusion_pred": pred
        })

    except Exception as e:
        print(f"❌ 错误: {img_path}: {e}")

# ✅ 保存结果
result_df = pd.DataFrame(results)
result_df.to_csv("fusion_test_result.csv", index=False)

# ✅ 简单准确率评估
acc = (result_df["fusion_pred"] == result_df["true_label"]).mean()
print(f"\n✅ 融合完成，共 {len(result_df)} 条样本，准确率：{acc:.4f}")

true_labels = result_df["true_label"].tolist()
pred_labels = result_df["fusion_pred"].tolist()

# ✅ 混淆矩阵
cm = confusion_matrix(true_labels, pred_labels)
print("📊 混淆矩阵:")
print(cm)

# ✅ 分类报告（可选）
print("\n📄 分类报告:")
print(classification_report(true_labels, pred_labels, target_names=["Negative", "Positive"]))

# ✅ 可视化
# ✅ 可视化并保存混淆矩阵
sns.heatmap(cm, annot=True, cmap="Greens", fmt="d",
            xticklabels=["Abnormal", "Normal"], yticklabels=["Abnormal", "Normal"])
plt.title("Test Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()

# ✅ 保存图片
plt.savefig("fusion_confusion_matrix.png", dpi=300)

# ✅ 显示图片
plt.show()
