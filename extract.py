import os
import numpy as np
import pandas as pd  

# ================= Configuration path =================
OFFICIAL_LIST_PATH = os.path.join('ImageList', 'Imagelist.txt')
TARGET_LIST_PATH = 'test_img.txt'
FEATURE_DIR = 'Low_Level_Features'
OUTPUT_DIR = 'Extracted_Features_Test' 

FEATURE_FILES = [
    'Normalized_CH.dat',
    'Normalized_CM55.dat',
    'Normalized_CORR.dat',
    'Normalized_EDH.dat',
    'Normalized_WT.dat',
    'BoW_int.dat'
]

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("1. Reading the official complete cast list and building a line number dictionary...")
    official_dict = {}
    with open(OFFICIAL_LIST_PATH, 'r') as f:
        for idx, line in enumerate(f):
            basename = os.path.basename(line.strip().replace('\\', '/'))
            official_dict[basename] = idx
            
    print(f"   -> Official list reading completed, total {len(official_dict)} images.")

    print("\n2. Reading your database_img.txt and searching for corresponding line numbers...")
    target_indices = []
    with open(TARGET_LIST_PATH, 'r') as f:
        for line in f:
            basename = os.path.basename(line.strip().replace('\\', '/'))
            if basename in official_dict:
                target_indices.append(official_dict[basename])
                
    print(f"   -> Your list requires extracting {len(target_indices)} images.")

    print("\n3. Starting feature matrix extraction (Pandas speed mode enabled)...\n")
    
    for feature_name in FEATURE_FILES:
        file_path = os.path.join(FEATURE_DIR, feature_name)
        print(f"   ⏳ Reading large file: {feature_name} (Please wait a few seconds)...")
        
        full_matrix = pd.read_csv(file_path, header=None, sep=r'\s+', engine='python').values
        
        print(f"   ✂️ Extracting {len(target_indices)} rows of data...")
        extracted_matrix = full_matrix[target_indices]
        
        save_name = feature_name.replace('.dat', '.npy')
        save_path = os.path.join(OUTPUT_DIR, save_name)
        np.save(save_path, extracted_matrix)
        
        print(f"   ✅ Extraction completed successfully! Saved as {save_path}, current matrix shape: {extracted_matrix.shape}\n")

    print(" All features have been extracted successfully!")

if __name__ == '__main__':
    main()