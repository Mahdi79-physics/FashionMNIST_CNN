import tensorflow as tf

import numpy as np

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

# Load Fashion MNIST dataset

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize pixel values

x_train = x_train.astype('float32') / 255.0

x_test = x_test.astype('float32') / 255.0

# Reshape for CNN input (add channel dimension)

x_train = x_train.reshape(-1, 28, 28, 1)

x_test = x_test.reshape(-1, 28, 28, 1)



# Split training data into train and validation (stratified)

x_train, x_val, y_train, y_val = train_test_split(

    x_train, y_train, test_size=0.2, stratify=y_train, random_state=42

)



# Sample one image from each class
fig, axes = plt.subplots(1, 10, figsize=(15, 5))
for i in range(10):
    class_indices = np.where(y_train == i)[0]
    sample_index = np.random.choice(class_indices)
    axes[i].imshow(X_train[sample_index].squeeze(), cmap='gray')
    axes[i].set_title(class_names[i])
    axes[i].axis('off')
plt.show()



class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',

               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']



plt.figure(figsize=(15, 5))

for i in range(10):

    plt.subplot(2, 5, i+1)

    idx = np.where(y_train == i)[0]

    index = np.random.choice(idx)

    plt.imshow(x_train[index].reshape(28, 28), cmap='gray')

    plt.title(class_names[i])

    plt.axis('off')

plt.show()



model = tf.keras.Sequential([
    # First Convolutional Block
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    # Second Convolutional Block
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    # Third Convolutional Block
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    
    # Flatten and Dense layers
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])



# Early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    x_train, y_train,
    epochs=50,
    batch_size=64,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping]
)




from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Predictions
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Print classification report
print(classification_report(y_test, y_pred_classes, target_names=class_names))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.show()


