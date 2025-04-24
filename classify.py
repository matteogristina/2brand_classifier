import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Uncomment if needed

import sys # Import the sys module to handle command-line arguments
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image

# --- Configuration ---
# Define the target image dimensions for ResNet
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Path to your saved model
model_path = "audi_bmw_resnet_final_224.keras"

# --- Argument Handling ---
# Check if the correct number of command-line arguments is provided
if len(sys.argv) != 2:
    # sys.argv[0] is the script name itself
    print(f"Usage: python3 {sys.argv[0]} <path_to_image_file>")
    sys.exit(1) # Exit with an error code

# Get the image path from the command-line argument
image_path = sys.argv[1]

# --- Load Model ---
try:
    model = load_model(model_path)
    print(f"Model '{model_path}' loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1) # Exit if model cannot be loaded

# --- Load and Preprocess Image ---
try:
    # Load the image using the path from the command line
    # Resize it to the target size (224x224)
    print(f"Loading image from: {image_path}")
    img = Image.open(image_path).convert('RGB')
    # Convert the image to a NumPy array and normalize pixel values to [0, 1]
    img = img.resize((IMG_HEIGHT, IMG_WIDTH)) 

    img_array = img_to_array(img)

    # Add a batch dimension (models expect input shape like: (batch_size, height, width, channels))
    img_array = np.expand_dims(img_array, axis=0) # Shape becomes (1, 224, 224, 3)
    img_processed = preprocess_input(img_array)
    print(f"Image '{image_path}' loaded and preprocessed.")

except FileNotFoundError:
    print(f"Error: Image file not found at '{image_path}'")
    sys.exit(1)
except Exception as e:
    print(f"Error processing image: {e}")
    sys.exit(1)

# --- Make Prediction ---
try:
    prediction = model.predict(img_array)
    print(f"Raw prediction output: {prediction}") # Show the raw output

    predictions_binary = (prediction > 0.5).astype(int).flatten()
    print(f"New output: {predictions_binary}") # Show the raw output

    # Interpret the prediction
    # Assuming the model outputs a single value (sigmoid activation):
    # Values < 0.5 -> Class 0 (likely 'Audi' if trained alphabetically)
    # Values >= 0.5 -> Class 1 (likely 'BMW' if trained alphabetically)
    # Adjust class names if your training labels were different.
    confidence = prediction[0][0]
    if confidence < 0.5:
        predicted_class = "Audi"
        confidence_percent = (1 - confidence) * 100
    else:
        predicted_class = "BMW"
        confidence_percent = confidence * 100

    print(f"Prediction: {predicted_class} (Confidence: {confidence_percent:.2f}%)")

except Exception as e:
    print(f"Error during prediction: {e}")
    sys.exit(1)

# Indicate successful completion (optional)
sys.exit(0)