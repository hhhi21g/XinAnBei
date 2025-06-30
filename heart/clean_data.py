import os
import glob
import numpy as np
import pandas as pd
import random
import soundfile as sf
import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import firwin, filtfilt, resample
from sklearn.model_selection import train_test_split


from polycoherence import polycoherence

# ================= 参数配置 =================
DATASET_ROOT = "xinan-wav"
CSV_SAVE_DIR = "output\\metadata"
NPY_SAVE_DIR = "output\\aug_data"
IMG_SAVE_DIR = "output\\bispec_output"
MAX_LEN = 16000 * 10
AUDIO_SR = 16000
IMG_SR = 1000
IMG_LEN = 3 * IMG_SR
IMG_SIZE = (4, 4)

def clear_folder(path):
    if os.path.exists(path):
        for f in os.listdir(path):
            fpath = os.path.join(path, f)
            if os.path.isfile(fpath) or os.path.islink(fpath):
                os.unlink(fpath)
            elif os.path.isdir(fpath):
                shutil.rmtree(fpath)
    else:
        os.makedirs(path)

for d in [CSV_SAVE_DIR, NPY_SAVE_DIR, IMG_SAVE_DIR]:
    clear_folder(d)

os.makedirs(CSV_SAVE_DIR, exist_ok=True)
os.makedirs(NPY_SAVE_DIR, exist_ok=True)
os.makedirs(IMG_SAVE_DIR, exist_ok=True)

# ================ 工具函数区域 ================
def bandpass_filter(signal, sr=16000, lowcut=20, highcut=400, numtaps=101):
    nyquist = 0.5 * sr
    taps = firwin(numtaps, [lowcut / nyquist, highcut / nyquist], pass_zero=False)
    return filtfilt(taps, [1.0], signal)

def resample_audio(y, orig_sr, target_sr):
    target_len = int(len(y) * target_sr / orig_sr)
    return resample(y, target_len)

def augment(waveform):
    if random.random() < 0.3:
        start = random.randint(0, len(waveform) - 1600)
        waveform[start:start + 1600] = 0
    if random.random() < 0.3:
        waveform += 0.005 * np.random.randn(len(waveform))
    if random.random() < 0.3:
        waveform *= random.uniform(0.8, 1.2)
    return waveform

def compute_bispectrum(y, sr=1000):
    y = y[:2500]
    f1, f2, bi_spectrum = polycoherence(y, fs=sr, nfft=1024, nperseg=256, noverlap=100, norm=None)
    bi_spectrum = np.abs(bi_spectrum)
    bi_spectrum = 255 * (bi_spectrum - bi_spectrum.min()) / (bi_spectrum.max() - bi_spectrum.min() + 1e-9)
    return bi_spectrum

def save_bispec_image(y, save_path, sr=1000):
    B = compute_bispectrum(y, sr)
    plt.figure(figsize=IMG_SIZE)
    plt.imshow(B, origin="lower", cmap="magma", aspect='auto')
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# ================ 加载全部路径与标签 ================
def load_all_data():
    dfs = []
    # === 1. 处理 Sejong, iStethoscope 等带有 REFERENCE.csv 的数据集 ===
    for csv_path in glob.glob(os.path.join(DATASET_ROOT, "**", "REFERENCE.csv"), recursive=True):
        base_dir = os.path.dirname(csv_path)
        df = pd.read_csv(csv_path, names=["id", "label"])
        df["path"] = df["id"].apply(lambda x: os.path.join(base_dir, f"{x}.wav"))
        df["target"] = df["label"].map({1: 1, -1: 0})
        dfs.append(df[["path", "target"]])

    # === 2. Yaseen 数据 ===
    yaseen_dir = os.path.join(DATASET_ROOT, "01_yaseen/01_yaseen")
    for label_name in os.listdir(yaseen_dir):
        folder = os.path.join(yaseen_dir, label_name)
        if os.path.isdir(folder):
            target = 0 if label_name.lower() == "normal" else 1
            for file in glob.glob(os.path.join(folder, "*.wav")):
                dfs.append(pd.DataFrame([{"path": file, "target": target}]))

    # === 3. BMD 数据 ===
    csv_path = os.path.join(DATASET_ROOT, "train.csv")
    audio_dir = os.path.join(DATASET_ROOT, "02_bmd/train")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            label = 0 if row['N'] == 1 else 1
            for i in range(1, 9):
                fname = row.get(f"recording_{i}")
                if pd.notna(fname):
                    path = os.path.join(audio_dir, fname)
                    if os.path.exists(path):
                        dfs.append(pd.DataFrame([{"path": path, "target": label}]))

    # === 4. PhysioNet 的 .aif 文件 ===
    for folder in ["Atraining_normal", "Btraining_normal"]:
        sub_dir = os.path.join(DATASET_ROOT, folder)
        if os.path.exists(sub_dir):
            for file in glob.glob(os.path.join(sub_dir, "*.aif")):
                dfs.append(pd.DataFrame([{"path": file, "target": 0}]))

    # ✅ 5. 新加入的 normal 文件夹结构处理
    normal_root = os.path.join(DATASET_ROOT, "normal")
    if os.path.exists(normal_root):
        for sub in os.listdir(normal_root):
            sub_dir = os.path.join(normal_root, sub)
            if os.path.isdir(sub_dir):
                for file in glob.glob(os.path.join(sub_dir, "*.wav")):
                    dfs.append(pd.DataFrame([{"path": file, "target": 0}]))

    return pd.concat(dfs, ignore_index=True)


# ================ 主处理流程：增强、保存 NPY 与图像 =================
def process_and_save(df, split_name):
    csv_save_path = os.path.join(CSV_SAVE_DIR, f"{split_name}_meta.csv")
    npy_save_dir = os.path.join(NPY_SAVE_DIR, split_name)
    img_save_dir = os.path.join(IMG_SAVE_DIR, split_name)
    os.makedirs(npy_save_dir, exist_ok=True)
    os.makedirs(img_save_dir, exist_ok=True)

    records = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"🔄 {split_name}"):
        try:
            y, sr = sf.read(row["path"])
            if y.ndim > 1: y = y[:, 0]
            if sr != AUDIO_SR: y = resample_audio(y, sr, AUDIO_SR)
            y = bandpass_filter(y)
            if len(y) > MAX_LEN:
                start = random.randint(0, len(y) - MAX_LEN)
                y = y[start:start + MAX_LEN]
            else:
                y = np.pad(y, (0, MAX_LEN - len(y)))
            y_aug = augment(y.copy())

            # 保存音频 .npy
            npy_path = os.path.join(npy_save_dir, f"sample_{i}.npy")
            np.save(npy_path, y_aug)

            # 保存图像 .png
            y_for_img = resample_audio(y, AUDIO_SR, IMG_SR)
            if len(y_for_img) > IMG_LEN:
                start = np.random.randint(0, len(y_for_img) - IMG_LEN)
                y_for_img = y_for_img[start:start + IMG_LEN]
            else:
                y_for_img = np.pad(y_for_img, (0, IMG_LEN - len(y_for_img)))

            label_dir = os.path.join(img_save_dir, str(row["target"]))
            os.makedirs(label_dir, exist_ok=True)
            img_path = os.path.join(label_dir, f"sample_{i}.png")
            save_bispec_image(y_for_img, img_path)

            records.append({"npy_path": npy_path, "img_path": img_path, "target": row["target"]})

        except Exception as e:
            print(f"❌ {row['path']} 失败，原因：{e}")

    pd.DataFrame(records).to_csv(csv_save_path, index=False)

# ================ 主程序入口 =================
if __name__ == "__main__":
    df_all = load_all_data()
    df_trainval, df_test = train_test_split(df_all, test_size=0.2, stratify=df_all["target"], random_state=42)
    df_train, df_val = train_test_split(df_trainval, test_size=0.125, stratify=df_trainval["target"], random_state=42)

    process_and_save(df_train, "train")
    process_and_save(df_val, "val")
    process_and_save(df_test, "test")

    print("✅ 完成！生成音频增强 NPY 与图像 PNG，并保存 CSV！")
