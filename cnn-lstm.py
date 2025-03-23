import os
import numpy as np
import seaborn as sns
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Dense, Dropout, Flatten, LSTM, GlobalAveragePooling2D)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from PIL import Image
import glob

# ================================
# Data Preparation
# ================================

data_dir = '/kaggle/input/driver-drowsiness-dataset-ddd/Driver Drowsiness Dataset (DDD)'
output_dir = '/kaggle/working/splitted_Data'

train_dir = os.path.join(output_dir, "train")
test_dir = os.path.join(output_dir, "test")
val_dir = os.path.join(output_dir, "val")

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_batches = train_datagen.flow_from_directory(train_dir, target_size=(128, 128), batch_size=16, class_mode='binary', shuffle=True)
test_batches = test_datagen.flow_from_directory(test_dir, target_size=(128, 128), batch_size=16, class_mode='binary', shuffle=False)
val_batches = val_datagen.flow_from_directory(val_dir, target_size=(128, 128), batch_size=16, class_mode='binary', shuffle=True)

# ================================
# Data Preprocessing
# ================================

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixels to [0,1]
    return img_array

# ================================
# Data Visualization
# ================================

def extract_class_name(path):
    return os.path.basename(os.path.dirname(path))

def show_images_with_labels(paths, num_images=15):
    plt.figure(figsize=(20, 10))
    for i, path in enumerate(paths[:num_images]):
        img = Image.open(path)
        label = extract_class_name(path)
        plt.subplot(3, 5, i + 1)
        plt.imshow(img)
        plt.title(label)
        plt.axis('off')
    plt.show()

train_paths = glob.glob(os.path.join(train_dir, "*/*.jpg"))  # Adjust based on actual dataset structure
show_images_with_labels(train_paths, num_images=15)

# ================================
# Model Definition (Pretrained CNN + LSTM)
# ================================

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False  # Freeze the pretrained layers

x = GlobalAveragePooling2D()(base_model.output)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.4)(x)

lstm_input = tf.keras.layers.Reshape((1, 128))(x)
lstm_output = LSTM(128, return_sequences=False)(lstm_input)
dense_output = Dense(1, activation='sigmoid')(lstm_output)

model = Model(inputs=base_model.input, outputs=dense_output)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# ================================
# Model Training
# ================================

history = model.fit(train_batches, validation_data=val_batches, epochs=20)

# ================================
# Model Evaluation
# ================================

y_true = test_batches.classes
y_pred = model.predict(test_batches)
y_pred = (y_pred > 0.5).astype(int).flatten()

print("\nClassification Report:")
print(classification_report(y_true, y_pred))

# ================================
# Confusion Matrix
# ================================

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Awake', 'Drowsy'], yticklabels=['Awake', 'Drowsy'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

