import numpy as np
import tensorflow as tf

print("--- Step 1 & 2: Loading CIFAR-100 dataset ---")
# Load fine labels (contains specific flowers like 'rose')
(x_train_fine, y_train_fine), (x_test_fine, y_test_fine) = tf.keras.datasets.cifar100.load_data(label_mode='fine')
# Load coarse labels (contains general categories like 'flowers')
(_, y_train_coarse), (_, y_test_coarse) = tf.keras.datasets.cifar100.load_data(label_mode='coarse')

print("--- Step 3, 4, 5, 6: Finding image indexes for FLOWERS (Coarse Label 2) ---")
# Find training and testing indexes where the coarse label equals 2 (Flowers)
train_indexes = np.where(y_train_coarse.flatten() == 2)[0]
test_indexes = np.where(y_test_coarse.flatten() == 2)[0]

print("--- Step 7 & 8: Extracting Flowers images and fine labels ---")
# Use the filtered indexes to pull out our sub-dataset
x_train_flowers = x_train_fine[train_indexes]
y_train_flowers = y_train_fine[train_indexes].flatten()

x_test_flowers = x_test_fine[test_indexes]
y_test_flowers = y_test_fine[test_indexes].flatten()

print("--- Step 9: Re-mapping fine labels to 0-4 scale ---")
# Map: 54->0 (orchid), 62->1 (poppy), 70->2 (rose), 82->3 (sunflower), 92->4 (tulip)
label_map = {54: 0, 62: 1, 70: 2, 82: 3, 92: 4}
y_train_flowers = np.array([label_map[label] for label in y_train_flowers])
y_test_flowers = np.array([label_map[label] for label in y_test_flowers])

print("\n================== VERIFICATION RESULTS ==================")
print(f"Training images shape: {x_train_flowers.shape} (Expected: (2500, 32, 32, 3))")
print(f"Training labels shape: {y_train_flowers.shape} (Expected: (2500,))")
print(f"Testing images shape:  {x_test_flowers.shape} (Expected: (500, 32, 32, 3))")
print(f"Testing labels shape:  {y_test_flowers.shape} (Expected: (500,))")
print(f"Unique fine labels mapped: {np.unique(y_train_flowers)} (Expected: [0 1 2 3 4])")
print("==========================================================")
