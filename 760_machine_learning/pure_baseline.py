import scipy.io as sio
import numpy as np
import warnings
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

warnings.filterwarnings('ignore')

# 1. 加载完整数据 (0% 缺失)
data = sio.loadmat('Caltech101-7.mat')
img_feat = data['X'][0][0]
txt_feat = data['X'][0][1]
true_labels = data['Y'].flatten()
num_clusters = len(np.unique(true_labels))

# 2. 传统方法：直接暴力拼接
fused_features = np.concatenate((img_feat, txt_feat), axis=1)

# 3. K-Means 聚类
kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
predicted_labels = kmeans.fit_predict(fused_features)

nmi_score = normalized_mutual_info_score(true_labels, predicted_labels)
print(f"🔥 数据 100% 完整时，传统 Baseline (暴力拼接) 的真实 NMI 得分: {nmi_score:.4f} 🔥")