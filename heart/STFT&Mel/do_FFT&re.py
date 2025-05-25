import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.signal import hilbert, butter, filtfilt

# 加载WAV文件（y: 音频信号，sr: 采样率）
y, sr = librosa.load("..\\dataSet\\a0001.wav", sr=None)
# y, sr = librosa.load("..\\dataSet\\a0003.wav", sr=None)
y = y[int(5 * sr):]
# y = librosa.resample(y, orig_sr=sr, target_sr=96000)
# sr = 96000
# 原始信号时长与采样点
N = len(y)
T = 1 / sr  # 采样间隔

# 傅里叶变换
Y = np.fft.fft(y)
freqs = np.fft.fftfreq(N, d=T)

# 创建掩码：仅保留50Hz±1Hz之间的频率
low_freq = 20500
high_freq = 22200
target_freq = 140.89
bandwidth = 100  # 频率容差
# mask = (abs(freqs - target_freq) < bandwidth)

mask = (np.abs(freqs) >= low_freq) & (np.abs(freqs) <= high_freq)

# 为保持共轭对称（实信号逆变换仍是实数），需对称频率一并保留
# mask |= (np.abs(freqs + target_freq) < bandwidth)

# 应用掩码
Y_filtered = Y * mask

# 反傅里叶变换
y_filtered = np.fft.ifft(Y_filtered).real  # 取实部

analytic_signal = hilbert(y_filtered)
envelope = np.abs(analytic_signal)

start = int((10 - 5) * sr)
end = int((12 - 5) * sr)
envelope = envelope[start:end]


def lowpass_filter(envelope, sr, cutoff=40):
    nyq = 0.5 * sr
    norm_cutoff = cutoff / nyq
    b, a = butter(N=4, Wn=norm_cutoff, btype='low')
    return filtfilt(b, a, envelope)


envelope_smooth = lowpass_filter(envelope, sr)
# 标准化包络
# envelope_norm = envelope_smooth / np.max(envelope_smooth + 1e-8)
envelope_log = np.log1p(envelope_smooth)
# 去掉直流偏置（整体背景噪声）
envelope_centered = envelope_log - np.median(envelope_log)
envelope_centered = np.clip(envelope_centered, a_min=0, a_max=None)
envelope_centered /= np.max(envelope_centered + 1e-8)


# y_seg = y[start:end]
# y_filtered_seg = y_filtered[start:end]
def moving_average(x, w=2000):
    return np.convolve(x, np.ones(w), 'same') / w


envelope_avg = moving_average(envelope_log, w=100)

plt.title("Filtered Signal")
librosa.display.waveshow(envelope_avg, sr=sr)
plt.tight_layout()
plt.show()
