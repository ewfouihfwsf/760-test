import scipy.io as sio
import numpy as np

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

# 1. 加载下载好的 mat 文件
data = sio.loadmat('Caltech101-7.mat')

# 2. 探查里面的内容 (通常学术界格式如下)
# X 通常是一个 cell 数组，X[0][0] 是图片特征，X[0][1] 是文本特征
# Y 通常是真实的标签

image_features = data['X'][0][0]  # 获取图片特征矩阵 (比如 2866 x 128)
text_features = data['X'][0][1]   # 获取文本特征矩阵 (比如 2866 x 100)
labels = data['Y']                # 获取真实标签 (用于最后算分)

print("图片特征形状:", image_features.shape)
print("文本特征形状:", text_features.shape)
print("标签形状:", labels.shape)



# ====== 第一步：搞破坏（模拟社交媒体缺失数据） ======
print("\n--- 开始模拟数据缺失 ---")
missing_rate = 0.30  # 设定 30% 的缺失率
num_samples = image_features.shape[0]

# 算出具体要毁掉多少个样本
num_missing = int(num_samples * missing_rate)
half_missing = num_missing // 2

# 复制一份数据，不要污染了原数据
img_missing = image_features.copy()
txt_missing = text_features.copy()

# 随机打乱这 1474 个帖子的索引
np.random.seed(42) # 固定随机种子，保证每次运行结果一样
indices = np.random.permutation(num_samples)

# 把选中的前一半帖子的图片变成全 0，后一半帖子的文本变成全 0
img_missing[indices[:half_missing]] = 0
txt_missing[indices[half_missing:num_missing]] = 0

print(f"成功将 {num_missing} 个样本的特征变成了全 0 (缺失率 {missing_rate*100}%)")


# ====== 第二步：基线模型登场（暴力拼接） ======
print("\n--- 运行基线方法 (Baseline) ---")
# 不管三七二十一，把图片(48维)和文本(40维)硬拼在一起，变成 88 维
combined_features = np.concatenate((img_missing, txt_missing), axis=1)
print(f"拼接后的特征形状: {combined_features.shape}")


# ====== 第三步：K-Means 无监督聚类 ======
# 先从真实标签里看看总共有几个类别 (比如 10 类还是 6 类)
true_labels = labels.flatten() # 把 (1474, 1) 压平变成一维数组
num_clusters = len(np.unique(true_labels))
print(f"数据集中共有 {num_clusters} 个真实类别，开始 K-Means 聚类...")

# 召唤 K-Means
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
predicted_labels = kmeans.fit_predict(combined_features)


# ====== 第四步：用真实标签打分 (NMI 指标) ======
nmi_score = normalized_mutual_info_score(true_labels, predicted_labels)
print(f"\n🔥🔥🔥 基线模型在 {missing_rate*100}% 缺失率下的最终得分 (NMI): {nmi_score:.4f} 🔥🔥🔥")