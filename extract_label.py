import os
import numpy as np

# ===== 1. 设置路径 =====
all_labels_dir = 'AllLabels'
imagelist_path = 'ImageList/Imagelist.txt'
db_img_path = 'database_img.txt'
test_img_path = 'test_img.txt'

# ===== 2. 读取官方 26万 总名单，提取纯文件名建立映射 =====
print(">>> 正在读取官方 Imagelist.txt...")
img_to_idx = {}
with open(imagelist_path, 'r') as f:
    for idx, line in enumerate(f):
        raw_name = line.strip().replace('\\', '/')
        # 核心修改：扒掉文件夹路径和 .jpg 后缀，只保留纯文件名 (如 "0001")
        base_name = os.path.splitext(os.path.basename(raw_name))[0]
        img_to_idx[base_name] = idx
print(f"共读取到 {len(img_to_idx)} 张原始图片名单。")

# ===== 3. 统计 81 个标签，自动选出包含图片最多的 Top 21 (TC21) =====
print("\n>>> 正在扫描 81 个原始标签文件，寻找 Top 21 类...")
label_files = [f for f in os.listdir(all_labels_dir) if f.endswith('.txt')]
concept_counts = []

for file in label_files:
    concept_name = file.replace('Labels_', '').replace('.txt', '')
    file_path = os.path.join(all_labels_dir, file)
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        count = sum(1 for line in lines if line.strip() == '1')
        concept_counts.append((concept_name, count, lines))

concept_counts.sort(key=lambda x: x[1], reverse=True)
top_21_concepts = concept_counts[:21]

# ===== 4. 提取标签并保存为 .npy =====
def extract_labels(target_img_path, output_npy_path):
    print(f"\n>>> 正在为 {target_img_path} 提取专属标签...")
    with open(target_img_path, 'r') as f:
        target_imgs = [line.strip().replace('\\', '/') for line in f]
    
    num_samples = len(target_imgs)
    label_matrix = np.zeros((num_samples, 21), dtype=np.int8)
    
    missing_count = 0
    for row_idx, img_name in enumerate(target_imgs):
        # 核心修改：目标名单也扒皮，只拿纯文件名去比对
        base_name = os.path.splitext(os.path.basename(img_name))[0]
        
        if base_name in img_to_idx:
            original_idx = img_to_idx[base_name]
            for col_idx, (_, _, lines) in enumerate(top_21_concepts):
                label_matrix[row_idx, col_idx] = int(lines[original_idx].strip())
        else:
            missing_count += 1
            
    print(f"✅ 提取完成！矩阵形状: {label_matrix.shape}")
    if missing_count > 0:
        print(f"⚠️ 警告: 依然有 {missing_count} 张图片未找到！(请检查名单)")
    else:
        print("🌟 完美匹配！没有遗漏任何图片！0 警告！")
        
    np.save(output_npy_path, label_matrix)
    print(f"💾 已成功保存为: {output_npy_path}")

# ===== 5. 运行提取 =====
extract_labels(db_img_path, 'database_labels.npy')
extract_labels(test_img_path, 'test_labels.npy')

print("\n🎉 大功告成！完美对齐的 labels 矩阵已经生成！")