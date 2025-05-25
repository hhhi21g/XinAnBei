import os
import pickle
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert
from sklearn.preprocessing import StandardScaler

# ========== 参数设置 ==========
input_dir = "..\\dataSet_original\\xyt"  # 音频文件夹
output_dir = ("..\\envelope_images\\3")  # 所有图像统一输出文件夹
os.makedirs(output_dir, exist_ok=True)

features_dir = "..\\features\\3"
os.makedirs(features_dir,exist_ok=True)

low_freq = 20500
high_freq = 21500
cutoff = 40
N_MFCC = 15


# ========== 工具函数 ==========
def lowpass_filter(signal, sr, cutoff=40):
    nyq = 0.5 * sr
    norm_cutoff = cutoff / nyq
    b, a = butter(4, norm_cutoff, btype='low')
    return filtfilt(b, a, signal)


user_mfcc_segments = []


def process_audio_file(file_path, output_dir):
    y, sr = librosa.load(file_path, sr=None)
    y = y[int(5 * sr):]  # 去掉前5秒

    # 高频滤波
    T = 1 / sr
    N = len(y)
    Y = np.fft.fft(y)
    freqs = np.fft.fftfreq(N, d=T)
    mask = (np.abs(freqs) >= low_freq) & (np.abs(freqs) <= high_freq)
    Y_filtered = Y * mask
    y_filtered = np.fft.ifft(Y_filtered).real

    # 包络提取与归一化
    envelope = np.abs(hilbert(y_filtered))
    envelope_smooth = lowpass_filter(envelope, sr, cutoff)
    envelope_log = np.log1p(envelope_smooth)
    envelope_centered = envelope_log - np.median(envelope_log)
    envelope_centered = np.clip(envelope_centered, a_min=0, a_max=None)
    envelope_centered /= np.max(envelope_centered + 1e-8)

    # 提取每秒并统一命名保存
    name_only = os.path.splitext(os.path.basename(file_path))[0]
    duration = int(len(envelope_centered) / sr)
    for sec in range(duration):
        start = int(sec * sr)
        end = int((sec + 1) * sr)
        if end > len(envelope_centered):
            break
        env_sec = envelope_centered[start:end]
        time_sec = np.linspace(0, 1, len(env_sec))

        plt.figure(figsize=(6, 2))
        plt.plot(time_sec, env_sec)
        plt.axis("off")  # 去除坐标轴
        save_path = os.path.join(output_dir, f"{name_only}_sec_{sec + 1:03d}.png")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        # 提取MFCC特征
        y_sec = y_filtered[start:end]
        try:
            mfcc = librosa.feature.mfcc(y=y_sec, sr=sr, n_mfcc=N_MFCC)
            mfcc_mean = np.mean(mfcc.T, axis=0)
            user_mfcc_segments.append(mfcc_mean)
        except Exception as e:
            print(f"❌ MFCC提取失败：{name_only}_sec_{sec + 1:03d}，错误：{e}")


# ========== 批量处理所有音频 ==========
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.wav', '.mp3', '.m4a')):
        file_path = os.path.join(input_dir, filename)
        print(f"🔄 正在处理：{filename}")
        process_audio_file(file_path, output_dir)

# ========== 保存用户特征档案 ==========
user_id = 3
user_mfcc_segments = np.array(user_mfcc_segments)

if user_mfcc_segments.shape[0] == 0:
    print(f"⚠️ 未提取到任何 MFCC 特征，跳过保存。请检查音频是否有效或处理是否出错。")
else:
    scaler = StandardScaler()
    user_mfcc_segments = scaler.fit_transform(user_mfcc_segments)

    # 保存为一个字典，包括特征和scaler
    data_to_save = {
        "features": user_mfcc_segments,
        "scaler": scaler
    }

    feature_filename = f"user_profile_{user_id}.pkl"
    feature_path = os.path.join(features_dir, feature_filename)
    with open(feature_path, "wb") as f:
        import pickle

        pickle.dump(data_to_save, f)

    print(f"✅ 用户 {user_id} 的 MFCC 特征提取完成，共 {len(user_mfcc_segments)} 条，已保存至：{feature_path}")
