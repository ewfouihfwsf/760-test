import numpy as np

# 咱们定义的 TC21 顺序
CLASSES = [
    'sky', 'clouds', 'person', 'water', 'animal', 
    'grass', 'buildings', 'window', 'plants', 'lake', 
    'ocean', 'road', 'flowers', 'sunset', 'reflection', 
    'rocks', 'vehicle', 'snow', 'tree', 'beach', 'mountain'
]

def verify(file_path):
    print(f"\n🔍 正在扫描文件: {file_path}")
    data = np.load(file_path)
    
    # 1. 检查是否全零
    total_ones = np.sum(data)
    if total_ones == 0:
        print("❌ 警告：该文件全是 0！说明一张图都没匹配上。")
        return
    
    # 2. 整体统计
    num_samples, num_classes = data.shape
    print(f"✅ 成功加载！样本数: {num_samples}, 类别数: {num_classes}")
    print(f"📊 整个矩阵共有 {total_ones} 个 '1'。")
    print(f"📈 平均每张图有 {total_ones/num_samples:.2f} 个标签。")

    # 3. 各类别分布统计 (查看每个类抓到了多少人)
    print("\n🚩 各类别命中统计:")
    class_counts = np.sum(data, axis=0) # 按列求和
    for i, count in enumerate(class_counts):
        print(f"   [{i+1:02d}] {CLASSES[i]:12} : {int(count):>6} 张图")

    # 4. 检查是否有异常值 (除了0和1以外的数)
    unique_vals = np.unique(data)
    if not np.array_equal(unique_vals, [0, 1]):
        print(f"🚨 警告：发现异常数值 {unique_vals}！标签应该只包含 0 和 1。")

# 运行检查
verify('database_labels.npy')
verify('test_labels.npy')