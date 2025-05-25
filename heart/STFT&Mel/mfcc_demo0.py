import os
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from demo1 import process_single_audio_file

# 所有用户特征路径
feature_base_dir = "..\\features"
test_audio_dir = "..\\testSet\\0"
model_save_path = "knn_user_model.pkl"
X = []
y = []

# 遍历每个用户的特征
for user_id in os.listdir(feature_base_dir):
    user_dir = os.path.join(feature_base_dir, user_id)
    if not os.path.isdir(user_dir):
        continue
    for filename in os.listdir(user_dir):
        if filename.endswith(".pkl"):
            filepath = os.path.join(user_dir, filename)
            with open(filepath, "rb") as f:
                data = pickle.load(f)
                mfcc_data = data["features"]
                scaler = data["scaler"]
                X.append(mfcc_data)
                y += [user_id] * len(mfcc_data)

# 拼接为 numpy 数组
X = np.vstack(X)
y = np.array(y)

# 数据中心化
mean = np.mean(X, axis=0)
X_centered = X - mean

# 奇异值分解
U, sigma, VT = np.linalg.svd(X_centered, full_matrices=False)

normalized_variances = (sigma ** 2) / np.sum(sigma ** 2)

# 舍弃前两个主成分
sum_first_two = np.sum(normalized_variances[:2])
sum_rest = 1.0 - sum_first_two
required_sum = sum_rest * 0.9  # 只保留其90%

selected_indices = []
current_sum = 0.0
for idx in range(2, len(normalized_variances)):
    current_sum += normalized_variances[idx]
    selected_indices.append(idx)
    if current_sum >= required_sum:
        break


# 所有样本进行投影
X_proj = (X - mean) @ VT.T[:, selected_indices]

# ================== 训练 KNN 模型 ==================
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_proj, y)

# ================== 保存模型 ==================
model = {
    "scaler":scaler,
    "mean": mean,
    "VT": VT,
    "selected_indices": selected_indices,
    "knn": knn
}
with open(model_save_path, "wb") as f:
    pickle.dump(model, f)

print("✅ 训练完成，模型已保存。开始对测试集进行预测...\n")

# ================== 测试集预测 ==================
with open(model_save_path, "rb") as f:
    model = pickle.load(f)

scaler = model["scaler"]
mean = model["mean"]
VT = model["VT"]
selected_indices = model["selected_indices"]
knn = model["knn"]

true_labels = []
pred_labels = []

for fname in os.listdir(test_audio_dir):
    if fname.lower().endswith(('.wav', '.m4a')):
        file_path = os.path.join(test_audio_dir, fname)
        mfcc_array = process_single_audio_file(file_path)
        if mfcc_array.shape[0] == 0:
            print(f"⚠️ 无有效 MFCC：{fname}")
            continue
        mfcc_scaled = scaler.transform(mfcc_array)
        # mfcc_centered = mfcc_array - mean
        projected = mfcc_scaled @ VT.T[:, selected_indices]
        test_vector = np.mean(projected, axis=0)

        pred = knn.predict([test_vector])[0]
        pred_labels.append(pred)

        # 假设文件名格式如：user1_record1.m4a，可提取真实标签
        true_label = fname.split("_")[0].replace("user", "")
        true_labels.append(true_label)

        print(f"🎧 {fname} → 预测：{pred}")

# ================== 输出整体结果 ==================
print("\n📊 测试集准确率：", accuracy_score(true_labels, pred_labels))
print(classification_report(true_labels, pred_labels))