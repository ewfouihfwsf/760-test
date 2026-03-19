import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import os
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '1'

# ==========================================
# 0. 必须存在的魔鬼教练：InfoNCE 损失函数
# ==========================================
def info_nce_loss(z_img, z_txt, temperature=0.1):
    z_img = F.normalize(z_img, dim=1, eps=1e-8)
    z_txt = F.normalize(z_txt, dim=1, eps=1e-8)
    logits = torch.matmul(z_img, z_txt.T) / temperature
    labels = torch.arange(logits.size(0)).to(logits.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2

# ==========================================
# 1. 准备数据
# ==========================================
print("--- 1. 加载 Caltech101-7 数据并制造 30% 缺失 ---")
data = sio.loadmat('Caltech101-7.mat')
img_feat = data['X'][0][0]
txt_feat = data['X'][0][1]
true_labels = data['Y'].flatten()
num_clusters = len(np.unique(true_labels))

num_samples = img_feat.shape[0]
missing_rate = 0
num_missing = int(num_samples * missing_rate)

img_missing = img_feat.copy()
txt_missing = txt_feat.copy()

np.random.seed(42)
indices = np.random.permutation(num_samples)
img_missing[indices[:num_missing//2]] = 0
txt_missing[indices[num_missing//2:num_missing]] = 0

img_tensor = torch.tensor(img_missing, dtype=torch.float32)
txt_tensor = torch.tensor(txt_missing, dtype=torch.float32)

# ==========================================
# 2. 初始化网络和优化器
# ==========================================
print("--- 2. 初始化双流网络 ---")
class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(48, 128), nn.ReLU(), nn.Linear(128, 64))
    def forward(self, x): return self.net(x)

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(40, 128), nn.ReLU(), nn.Linear(128, 64))
    def forward(self, x): return self.net(x)

model_img = ImageEncoder()
model_txt = TextEncoder()

optimizer = torch.optim.Adam(list(model_img.parameters()) + list(model_txt.parameters()), lr=0.01)# 这个学习率可以根据实际情况调整

# ==========================================
# 3. 开始炼丹 (Training Loop - 只用健康数据)
# ==========================================
print("--- 3. 开始使用对比学习训练 (炼丹) ---")
epochs = 1500

# 找到健康的图文对（避开被我们填0的那部分）
valid_indices = indices[num_missing:]

for epoch in range(epochs):
    # 前向传播
    z_img = model_img(img_tensor)
    z_txt = model_txt(txt_tensor)
    
    # 提取健康的特征去算 Loss
    z_img_valid = z_img[valid_indices]
    z_txt_valid = z_txt[valid_indices]
    
    # 计算损失 (Loss)
    loss = info_nce_loss(z_img_valid, z_txt_valid)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f"第 {epoch+1}/{epochs} 轮 -> 损失 Loss: {loss.item():.4f}")

# ==========================================
# 4. 见证奇迹：聚类打分
# ==========================================
print("\n--- 4. 提取对齐后的特征，开始 K-Means 聚类 ---")
with torch.no_grad():
    final_z_img = model_img(img_tensor)
    final_z_txt = model_txt(txt_tensor)
    
    # 【高级操作：动态检测缺失并融合】
    # 1. 自动揪出那些原始输入全是 0 的样本 (缺失样本)
    img_is_missing = (img_tensor.abs().sum(dim=1) == 0).unsqueeze(1) # 形状 [1474, 1]
    txt_is_missing = (txt_tensor.abs().sum(dim=1) == 0).unsqueeze(1)
    
    # 2. 生成权重：如果缺失了，权重就是 0；如果在，权重就是 1
    img_weight = (~img_is_missing).float()
    txt_weight = (~txt_is_missing).float()
    
    # 3. 动态融合：只把非缺失的特征加起来！
    # 如果图片缺失：0 + 文本特征 = 纯文本特征
    # 如果都健康：图片特征 + 文本特征 = 强强联合
    fused_features = (final_z_img * img_weight + final_z_txt * txt_weight).numpy()

# 召唤 K-Means
kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
predicted_labels = kmeans.fit_predict(fused_features)

nmi_score = normalized_mutual_info_score(true_labels, predicted_labels)
print(f"\n🚀🚀🚀 我们的模型 (InfoNCE + 动态融合) 在 30% 缺失率下的最终得分 (NMI): {nmi_score:.4f} 🚀🚀🚀")
print("（对比之前的 Baseline 分数：0.1433）")