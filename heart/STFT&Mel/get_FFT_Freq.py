import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.signal import find_peaks

# 加载音频并去除前5秒
y, sr = librosa.load("..\\dataSet_original\\lhb\\record15.m4a", sr=None)
y = y[int(5 * sr):]
y = y - np.mean(y)
# 参数
N = len(y)
T = 1 / sr

# 傅里叶变换
Y = np.fft.fft(y)
freqs = np.fft.fftfreq(N, d=T)

# 只取正频率部分
half_N = N // 2
Y_half = Y[:half_N]
freqs_half = freqs[:half_N]
magnitude = np.abs(Y_half)

# 寻找幅度谱中的峰值
peaks, _ = find_peaks(magnitude, height=np.max(magnitude) * 0.2, distance=20)

# 提取峰值对应频率和幅值
peak_freqs = freqs_half[peaks]
peak_magnitudes = magnitude[peaks]

# ✅ 打印峰值频率与幅度
print("🔍 检测到的峰值频率与幅度如下：\n")
for i, (f, a) in enumerate(zip(peak_freqs, peak_magnitudes)):
    print(f"🔹 峰值 {i+1:>2}: 频率 = {f:>7.2f} Hz, 幅度 = {a:>8.2f}")

# 可视化完整频谱与峰值标注
plt.figure(figsize=(12, 5))
plt.plot(freqs_half, magnitude, label='Frequency Spectrum')
plt.plot(peak_freqs, peak_magnitudes, "x", label='Peaks', color='red')
for f, a in zip(peak_freqs, peak_magnitudes):
    plt.text(f, a, f"{f:.2f}Hz", fontsize=8, rotation=45)
plt.title("Peak Frequencies in Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
