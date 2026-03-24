import numpy as np
import os

# ===== 1. 定义类别名称 (按咱们刚才跑出的 Top 21 顺序) =====
CLASSES = [
    'sky', 'clouds', 'person', 'water', 'animal', 
    'grass', 'buildings', 'window', 'plants', 'lake', 
    'ocean', 'road', 'flowers', 'sunset', 'reflection', 
    'rocks', 'vehicle', 'snow', 'tree', 'beach', 'mountain'
]

def check_file(name, path):
    if not os.path.exists(path):
        print(f"❌ 找不到文件: {path}")
        return
    
    data = np.load(path)
    print(f"\n--- 📄 文件: {name} ---")
    print(f"📍 存储路径: {path}")
    print(f"📊 矩阵形状: {data.shape}")
    print(f"🧬 数据类型: {data.dtype}")
    
    # 统计数值信息
    if np.issubdtype(data.dtype, np.floating):
        print(f"📈 数值范围: [{data.min():.4f}, {data.max():.4f}]")
        print(f"平均值: {data.mean():.4f} | 标准差: {data.std():.4f}")
    
    # 检查是否有无效值
    if np.isnan(data).any():
        print("🚨 警告: 包含 NaN (空值)！")
    
    # 如果是标签文件，随机抽查语义
    if 'label' in name.lower():
        pos_count = np.sum(data)
        print(f"✅ 标签总数 (1的个数): {pos_count}")
        print(f"每张图平均标签数: {pos_count / data.shape[0]:.2f}")
        
        # 随机抽一张有标签的图看看
        print("💡 随机抽样检查 (索引 100):")
        sample_labels = data[100]
        active_classes = [CLASSES[i] for i, val in enumerate(sample_labels) if val == 1]
        print(f"   该图所属类别: {active_classes if active_classes else '无标签'}")

# ===== 2. 执行检查 =====
print("🚀 开始数据全家桶体检...")

# 检查数据库部分 (Database)
check_file("训练标签", "database_labels.npy")
# 这里你可以换成你提取的任何一个特征文件名，比如 Normalized_CH.npy
check_file("图像颜色特征", "Normalized_CH.npy") 

# 检查测试集部分 (Test)
check_file("测试标签", "test_labels.npy")

print("\n\n✅ 如果维度(Shape)对齐，且没有 NaN，你的数据就可以起飞了！")