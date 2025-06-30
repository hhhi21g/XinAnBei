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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ EfficientNet 加载
efficientnet = efficientnet_b1(weights=None)  # ⚠️ 一定是 b1，不是 b0
efficientnet.classifier[1] = nn.Linear(efficientnet.classifier[1].in_features, 1)
# 加载权重
efficientnet.load_state_dict(torch.load("three_models\\best_model.pth", map_location=DEVICE))
efficientnet.to(DEVICE).eval()

# ✅ ConvNeXt-Tiny 加载（num_classes=2）
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

# ✅ 音频处理器
processor = Wav2Vec2Processor.from_pretrained("wav2vec2-base")

# ✅ 图像预测
def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        p1 = torch.sigmoid(efficientnet(img)).squeeze().item()
        p2 = F.softmax(convnext(img), dim=1)[:, 1].squeeze().item()
    return p1, p2

# ✅ 音频预测
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
        p3 = wav_model(input_values).item()
    return p3

# ✅ 加载元数据
meta_csv = "output\\metadata\\test_meta.csv"
df = pd.read_csv(meta_csv)

results = []

print("🔍 正在执行融合推理...")
for _, row in tqdm(df.iterrows(), total=len(df)):
    img_path = row["img_path"]
    wav_path = row["npy_path"]
    target = row["target"]

    if not os.path.exists(img_path) or not os.path.exists(wav_path):
        print(f"⚠️ 文件缺失: {img_path} 或 {wav_path}")
        continue

    try:
        p1, p2 = predict_image(img_path)
        p3 = predict_audio(wav_path)

        votes = [int(p1 >= 0.5), int(p2 >= 0.5), int(p3 >= 0.5)]
        fusion_pred = int(sum(votes) >= 2)
        avg_prob = np.mean([p1, p2, p3])

        results.append({
            "img_path": img_path,
            "npy_path": wav_path,
            "target": target,
            "p1": round(p1, 4),
            "p2": round(p2, 4),
            "p3": round(p3, 4),
            "avg_prob": round(avg_prob, 4),
            "fusion_pred": fusion_pred
        })

    except Exception as e:
        print(f"❌ 错误: {img_path}: {e}")

# ✅ 保存融合结果 CSV
result_df = pd.DataFrame(results)
result_df.to_csv("fusion_test_result.csv", index=False)
print(f"✅ 已完成融合推理，共保存 {len(result_df)} 条结果至 fusion_test_result.csv")
