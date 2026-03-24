import numpy as np

# 1. Load the CH (color histogram) features we just generated.
npy_path = 'Extracted_Features/Normalized_CH.npy'
print(f"Loading {npy_path} ...")
my_features = np.load(npy_path)

# 2. First step in inspection: Check the shape (Shape) is correct?
print("\n--- [First step in inspection: Shape check] ---")
print(f"The shape of the matrix is: {my_features.shape}")
print("Theoretically, the number of rows should equal the number of images in your database_img.txt (193734).")
print("The number of columns should equal 64 (since CH features are 64-dimensional).")

# 3. First step in inspection: Explore the data to see if it's normal floating-point numbers?
print("\n--- [Second step in inspection: Data exploration] ---")
print("Let's take a look at the first 5 feature values of the first image:")
print(my_features[0][:5]) # Print the first row, first 5 numbers

# 4. First step in inspection: Check the dimensionality of BoW_int?
bow_path = 'Extracted_Features/BoW_int.npy'
bow_features = np.load(bow_path)
print("\n--- [Third step in inspection: BoW dimension check] ---")
print(f"BoW matrix shape is: {bow_features.shape} (column count must be 500)")

# This code is for you to check if the extracted features are in the correct shape and format(These codes are for Extracted_Features not for Extracted_Features_Test).