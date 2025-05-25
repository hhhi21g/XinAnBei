import pickle

import librosa
import numpy as np


def extract_mfcc_from_file(file_path, n_mfcc=15):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc.T  # 每帧是一个 MFCC 特征向量

def predict_knn_identity(mfcc_feat, model_path="knn_user_model.pkl"):
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    mean = model["mean"]
    VT = model["VT"]
    selected_indices = model["selected_indices"]
    knn = model["knn"]

    # 特征中心化 + 投影
    feat_centered = mfcc_feat - mean
    projected = feat_centered @ VT.T[:, selected_indices]
    test_profile = np.mean(projected, axis=0)

    # 使用 KNN 预测
    pred = knn.predict([test_profile])[0]
    print(f"✅ 预测身份为：用户 {pred}")
    return pred

mfcc_feat = extract_mfcc_from_file("../data/survey1/record_generate2_25.wav")
predict_knn_identity(mfcc_feat)
