from sklearn.metrics import classification_report

# 示例真实标签（1表示异常，0表示正常）
# ✅ 构造真实标签与预测标签
y_true = [1]*535 + [0]*346            # 实际标签（1表示异常，0表示正常）
y_pred = [1]*506 + [0]*29 + [1]*60 + [0]*286  # 预测标签，顺序与混淆矩阵一致

# 生成报告
target_names = ["Normal", "Abnormal"]
report = classification_report(y_true, y_pred, target_names=target_names)

# 打印结果
print("📋 Classification Report:")
print(report)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

# ✅ 混淆矩阵数值
cm = np.array([[506, 35],
               [47, 286]])

# ✅ 类别标签
labels = ['Abnormal', 'Normal']

# ✅ 画图
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Abnormal', 'Normal'],
            yticklabels=['Abnormal', 'Normal'])

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()
