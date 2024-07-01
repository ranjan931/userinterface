import streamlit as st
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

st.title('MediScan: AI-Powered Medical Image Analysis for Disease Diagnosis')
st.write("This application helps in diagnosing eye diseases using AI.")

# Upload kaggle.json
st.sidebar.title("Kaggle API Key")
uploaded_file = st.sidebar.file_uploader("Upload your kaggle.json file", type="json")

if uploaded_file:
    with open('kaggle.json', 'wb') as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success('Uploaded kaggle.json')

    # Configure Kaggle API
    os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
    os.rename('kaggle.json', os.path.expanduser('~/.kaggle/kaggle.json'))
    os.chmod(os.path.expanduser('~/.kaggle/kaggle.json'), 0o600)

    # Download dataset
    st.sidebar.write("Downloading dataset...")
    os.system('kaggle datasets download -d gunavenkatdoddi/eye-diseases-classification')
    os.system('unzip -q eye-diseases-classification.zip -d dataset')

    st.sidebar.success('Dataset downloaded and extracted')

    # Verify dataset directory
    if not os.path.exists('dataset'):
        st.error('Dataset directory not found after extraction')
    else:
        st.success('Dataset directory found and ready for processing')

# Preprocess Images
st.header('Preprocessing Images')
rescale = tf.keras.layers.Rescaling(1./255)

# Load train dataset
if os.path.exists('dataset'):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory='dataset',
        batch_size=32,
        image_size=(256, 256),
        validation_split=0.2,
        subset="training",
        seed=123,
        label_mode='categorical',
    )
    train_ds = train_ds.map(lambda x, y: (rescale(x), y))
    validation_ds = tf.keras.utils.image_dataset_from_directory(
        directory='dataset',
        batch_size=32,
        image_size=(256, 256),
        validation_split=0.2,
        subset="validation",
        seed=123,
        label_mode='categorical',
    )
    validation_ds = validation_ds.map(lambda x, y: (rescale(x), y))
    test_ds = tf.keras.utils.image_dataset_from_directory(
        directory='dataset',
        batch_size=32,
        image_size=(256, 256),
        label_mode='categorical',
        shuffle=False,
    )
    test_ds = test_ds.map(lambda x, y: (rescale(x), y))

    st.success('Image datasets loaded successfully')
else:
    st.error('Dataset directory not found. Please upload the kaggle.json file and download the dataset.')

# Visualize Sample Images
def visualize_images(path, target_size=(256, 256), num_images=5):
    image_filenames = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    if not image_filenames:
        raise ValueError("No images found in the specified path")

    selected_images = random.sample(image_filenames, min(num_images, len(image_filenames)))
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3), facecolor='white')

    for i, image_filename in enumerate(selected_images):
        image_path = os.path.join(path, image_filename)
        image = Image.open(image_path)
        image = image.resize(target_size)
        axes[i].imshow(image)
        axes[i].axis('off')
        axes[i].set_title(image_filename)

    plt.tight_layout()
    st.pyplot(fig)

if os.path.exists('dataset'):
    st.header('Sample Images')
    path_to_visualize = st.selectbox('Select Category', ['dataset/cataract', 'dataset/glaucoma', 'dataset/diabetic_retinopathy', 'dataset/normal'])
    visualize_images(path_to_visualize, num_images=5)
else:
    st.error('Dataset directory not found. Please upload the kaggle.json file and download the dataset.')

# Build and Train the Model
st.header('Model Training')

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Reshape((64, 1)),
    tf.keras.layers.SimpleRNN(32),
    tf.keras.layers.Dense(4, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

st.write(model.summary())

# Train the model
if st.button('Train Model'):
    if os.path.exists('dataset'):
        with st.spinner('Training...'):
            history = model.fit(
                train_ds,
                validation_data=validation_ds,
                epochs=5,
            )
        st.success('Training completed')
        st.write(history.history)
    else:
        st.error('Dataset directory not found. Please upload the kaggle.json file and download the dataset.')

# Test the Model and Display Results
st.header('Model Evaluation')

# Evaluate the model
if st.button('Evaluate Model'):
    if os.path.exists('dataset'):
        with st.spinner('Evaluating...'):
            test_loss, test_acc = model.evaluate(test_ds)
        st.write(f'Test Accuracy: {test_acc}')
        st.write(f'Test Loss: {test_loss}')

        # Generate classification report
        y_pred = np.argmax(model.predict(test_ds), axis=-1)
        y_true = np.concatenate([y for x, y in test_ds], axis=0)
        y_true = np.argmax(y_true, axis=1)
        report = classification_report(y_true, y_pred, target_names=['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal'])
        st.text(report)
    else:
        st.error('Dataset directory not found. Please upload the kaggle.json file and download the dataset.')

