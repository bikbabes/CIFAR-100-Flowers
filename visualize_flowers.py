import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

print("--- Loading and filtering dataset for visualization ---")
(x_train_fine, y_train_fine), (_, _) = tf.keras.datasets.cifar100.load_data(label_mode='fine')
(_, y_train_coarse), (_, _) = tf.keras.datasets.cifar100.load_data(label_mode='coarse')

# Filter for Flowers coarse class (label 2)
train_indexes = np.where(y_train_coarse.flatten() == 2)
x_train = x_train_fine[train_indexes]
y_train = y_train_fine[train_indexes].flatten()

# Class names in order of their mapped labels (0=orchid, 1=poppy, 2=rose, 3=sunflower, 4=tulip)
flower_names = ['Orchid', 'Poppy', 'Rose', 'Sunflower', 'Tulip']
fine_labels = [54, 62, 70, 82, 92]

# Set up a wide window to display 5 images side-by-side
plt.figure(figsize=(10, 3))

print("--- Finding sample images ---")
for i, label in enumerate(fine_labels):
    # Find the index of the first image that matches this flower type
    img_idx = np.where(y_train == label)[0][0]
    img = x_train[img_idx]
    
    # Add it to our grid layout
    plt.subplot(1, 5, i + 1)
    plt.imshow(img)
    plt.title(flower_names[i])
    plt.axis('off')  # Hide grid axis lines

print("--- Opening display window ---")
plt.tight_layout()
plt.show()
