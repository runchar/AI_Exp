import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 1. 读取数据
label_path = r'../output/exp3_0601/labels.npy'
pred_path = r'../output/exp3_0601/predicted.npy'

if not os.path.exists(label_path):
    print(f"标签文件不存在: {label_path}")
    exit(1)
if not os.path.exists(pred_path):
    print(f"预测文件不存在: {pred_path}")
    exit(1)

labels = np.load(label_path)
preds = np.load(pred_path)

# 2. 计算指标
acc = accuracy_score(labels, preds)
prec = precision_score(labels, preds, average='macro')
rec = recall_score(labels, preds, average='macro')
f1 = f1_score(labels, preds, average='macro')

print(f'准确率: {acc:.4f}')
print(f'精确率: {prec:.4f}')
print(f'召回率: {rec:.4f}')
print(f'F1分数: {f1:.4f}')

# 3. 计算混淆矩阵
cm = confusion_matrix(labels, preds)
print('混淆矩阵:')
print(cm)


# 5. 绘制并保存混淆矩阵图像
import matplotlib
matplotlib.use('Agg')  # 添加此行，避免 Qt 报错
import matplotlib.pyplot as plt


plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(np.unique(labels)))
plt.xticks(tick_marks, tick_marks)
plt.yticks(tick_marks, tick_marks)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.grid(False)
plt.tick_params(axis='both', which='both', length=0)
plt.gca().spines[:].set_visible(False)
plt.gcf().set_dpi(400)

# 在图像上标注数字
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.savefig(r'../output/exp3_0601/confusion_matrix.png')
plt.close()
