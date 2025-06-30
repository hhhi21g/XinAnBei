import os
import glob
import numpy as np
import pandas as pd
import soundfile as sf
import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import firwin, filtfilt, resample

# ====== 参数配置 ======
DATASET_ROOT = ("xinan-wav")  # ✅ 你的测试音频文件夹
CSV_SAVE_DIR = "output\\metadata"
NPY_SAVE_DIR = "output\\aug_data"
IMG_SAVE_DIR = "output\\bispec_output"
MAX_LEN = 16000 * 10
AUDIO_SR = 16000
IMG_SR = 1000
IMG_LEN = 3 * IMG_SR
IMG_SIZE = (4, 4)

# ====== 加载 polycoherence.py（自定义）======
from polycoherence import polycoherence

# ====== 文件夹清理函数 ======
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

# ====== 滤波与重采样函数 ======
def bandpass_filter(signal, sr=16000, lowcut=20, highcut=400, numtaps=101):
    nyquist = 0.5 * sr
    taps = firwin(numtaps, [lowcut / nyquist, highcut / nyquist], pass_zero=False)
    return filtfilt(taps, [1.0], signal)

def resample_audio(y, orig_sr, target_sr):
    target_len = int(len(y) * target_sr / orig_sr)
    return resample(y, target_len)

# ====== 二阶谱图图像保存 ======
def compute_bispectrum(y, sr=1000):
    y = y[:1500]
    f1, f2, bi_spectrum = polycoherence(y, fs=sr, nfft=256, nperseg=128, noverlap=64, norm=None)
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

# ====== 加载测试音频路径 ======
def load_test_folder(folder_path):
    audio_paths = glob.glob(os.path.join(folder_path, "*.wav"))
    return pd.DataFrame({
        "path": audio_paths,
        "target": [-1] * len(audio_paths)  # 用 -1 占位，表示未知标签
    })

# ====== 主处理函数（无增强） ======
def process_and_save(df, split_name):
    csv_save_path = os.path.join(CSV_SAVE_DIR, f"{split_name}_meta.csv")
    npy_save_dir = os.path.join(NPY_SAVE_DIR, split_name)
    img_save_dir = os.path.join(IMG_SAVE_DIR, split_name)
    os.makedirs(npy_save_dir, exist_ok=True)
    os.makedirs(img_save_dir, exist_ok=True)

    records = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"🔍 Processing {split_name}"):
        try:
            y, sr = sf.read(row["path"])
            if y.ndim > 1: y = y[:, 0]
            if sr != AUDIO_SR: y = resample_audio(y, sr, AUDIO_SR)
            y = bandpass_filter(y)
            if len(y) > MAX_LEN:
                start = np.random.randint(0, len(y) - MAX_LEN)
                y = y[start:start + MAX_LEN]
            else:
                y = np.pad(y, (0, MAX_LEN - len(y)))

            # 保存 .npy
            npy_path = os.path.join(npy_save_dir, f"sample_{i}.npy")
            np.save(npy_path, y)

            # 保存谱图
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

# ====== 入口点 ======
if __name__ == "__main__":
    df_test = load_test_folder(DATASET_ROOT)
    process_and_save(df_test, "test")
    print("✅ 已完成测试音频的 .npy 和图像预处理，无增强")
