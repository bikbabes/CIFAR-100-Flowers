import numpy as np
import tensorflow as tf

print("--- Step 1: Loading and filtering the Flowers dataset ---")
# Load fine and coarse labels
(x_train_fine, y_train_fine), (x_test_fine, y_test_fine) = tf.keras.datasets.cifar100.load_data(label_mode='fine')
(_, y_train_coarse), (_, y_test_coarse) = tf.keras.datasets.cifar100.load_data(label_mode='coarse')

# Filter for Flowers coarse class (label 2)
train_indexes = np.where(y_train_coarse.flatten() == 2)
test_indexes = np.where(y_test_coarse.flatten() == 2)

x_train = x_train_fine[train_indexes]
y_train = y_train_fine[train_indexes].flatten()
x_test = x_test_fine[test_indexes]
y_test = y_test_fine[test_indexes].flatten()

# Re-map fine labels to 0-4 scale
label_map = {54: 0, 62: 1, 70: 2, 82: 3, 92: 4}
y_train = np.array([label_map[label] for label in y_train])
y_test = np.array([label_map[label] for label in y_test])

print("--- Step 2: Normalizing pixel values ---")
# Convert pixel integers (0-255) to decimals (0.0-1.0) so the neural network learns faster
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

print("--- Step 3: Designing the Convolutional Neural Network (CNN) ---")
model = tf.keras.models.Sequential([
    # First convolution layer to spot patterns (edges, shapes)
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    # Second convolution layer for more complex patterns
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    # Flatten the image data into a single 1D list
    tf.keras.layers.Flatten(),
    
    # Dense hidden layer with 64 nodes
    tf.keras.layers.Dense(64, activation='relu'),
    
    # Output layer with 5 nodes (one for each flower class: orchid, poppy, rose, sunflower, tulip)
    tf.keras.layers.Dense(5, activation='softmax')
])

print("--- Step 4: Compiling the model ---")
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("--- Step 5: Training the neural network (3 epochs) ---")
# We use 3 epochs (loops through the data) so you can verify it works quickly without waiting all day
model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))

print("\n--- Step 6: Evaluating on test data ---")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nFinal Test Accuracy: {test_acc * 100:.2f}%")
