import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split

def load_data(dataset_path):
    images, labels = [], []
    class_labels = ["glacier", "mountain", "forest", "buildings", "street", "sea"]
    
    print("Loading data...")
    for i, folder in enumerate(class_labels):
        folder_path = os.path.join(dataset_path, folder)
        
        for file in tqdm(os.listdir(folder_path), desc=f"Processing {folder}"):
            img_path = os.path.join(folder_path, file)
            
            image = Image.open(img_path).resize((150, 150))
            images.append(np.array(image))
            labels.append(i)
            
    images = np.stack(images)
    labels = np.array(labels, dtype='int32')
    
    return images, labels

def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

def train():
    # Define your data path
    dataset_path = r"C:\Users\aolay\OneDrive\Documents\MIDS\DataSci 281\Final Project\seg_train"
    
    # Load data
    images, labels = load_data(dataset_path)
    
    # Convert labels to one-hot encoding
    num_classes = 6  # number of your classes
    labels_onehot = tf.keras.utils.to_categorical(labels, num_classes)
    
    # Create train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        images, 
        labels_onehot,
        test_size=0.2,
        random_state=42
    )
    
    # Normalize images
    X_train = X_train / 255.0
    X_val = X_val / 255.0
    
    # Compile the CNN model
    cnn_model = create_cnn_model((150, 150, 3), num_classes)
    cnn_model.compile(optimizer='adam', 
                     loss='categorical_crossentropy', 
                     metrics=['accuracy'])

    # Train the CNN Model
    cnn_history = cnn_model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=10,
        validation_data=(X_val, y_val)
    )

    # Create model directory in the deployment folder
    model_dir = r"C:\Users\aolay\OneDrive\Documents\MIDS\DataSci Portfolio\Projects\Deploy Image Classification\deploy_image_classification\image_classification\model"
    os.makedirs(model_dir, exist_ok=True)

    # Save the trained model
    model_path = os.path.join(model_dir, 'cnn_model.h5')
    cnn_model.save(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train()