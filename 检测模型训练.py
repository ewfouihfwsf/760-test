import numpy as np

# ===== 1. 加载所有特征 (X) =====
print(">>> 正在加载图像特征...")
# 这里的路径我写了相对路径，如果在 Colab 跑，替换为他的 "/content/drive/..."
# 如果你在本地跑，确保这些特征文件在同级目录的 Extracted_Features 文件夹下
try:
    BoW_int = np.load("Extracted_Features/BoW_int.npy")
    Nor_CH = np.load("Extracted_Features/Normalized_CH.npy")
    Nor_CM = np.load("Extracted_Features/Normalized_CM55.npy")
    Nor_CO = np.load("Extracted_Features/Normalized_CORR.npy")
    Nor_EDH = np.load("Extracted_Features/Normalized_EDH.npy")
    Nor_WT = np.load("Extracted_Features/Normalized_WT.npy")
except FileNotFoundError as e:
    print(f"❌ 找不到特征文件，请检查路径: {e}")
    exit()

# 确保 6 个特征的行数（样本数）完全一致
assert BoW_int.shape[0] == Nor_CH.shape[0] == Nor_CM.shape[0] == Nor_CO.shape[0] == Nor_EDH.shape[0] == Nor_WT.shape[0]

# ⚠️ 优化小细节：BoW 是整数，其他是 0~1 的小数。为了防止拼接后数据类型混乱，统一转为 float32
BoW_float = BoW_int.astype(np.float32)

# 将 6 种特征横向拼接 (axis=1) 融合成一个超长的向量
input_features = np.concatenate([BoW_float, Nor_CH, Nor_CM, Nor_CO, Nor_EDH, Nor_WT], axis=1)
print(f"✅ 特征拼接完成！总特征矩阵 (X) 维度: {input_features.shape}，数据类型: {input_features.dtype}")


# ===== 2. 加载完美清洗的标签 (Y) =====
print("\n>>> 正在加载 TC-21 黄金标签...")
# 直接读取咱们做好的 .npy，彻底抛弃原来极易出错的 txt 读取和切片！
try:
    # 同样，在 Colab 跑记得改路径
    labels = np.load("database_labels.npy").astype(np.float32) 
    print(f"✅ 标签加载完成！标签矩阵 (Y) 维度: {labels.shape}")
except FileNotFoundError:
    print("❌ 找不到 database_labels.npy，请确保该文件在当前目录下！")
    exit()


# ===== 3. 终极对齐检查 =====
print("\n>>> 正在进行最终的特征与标签对齐校验...")
if input_features.shape[0] == labels.shape[0]:
    print(f"🎉 完美匹配！特征行数 ({input_features.shape[0]}) 与 标签行数 ({labels.shape[0]}) 100% 一致！")
    print("🚀 现在的 input_features 和 labels 已经可以直接送入 PyTorch 模型进行训练了！")
else:
    print(f"🚨 致命错误：特征行数 ({input_features.shape[0]}) 和 标签行数 ({labels.shape[0]}) 不一致，张冠李戴了！")