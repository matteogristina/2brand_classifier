import os
# Ensure TensorFlow uses CPU if GPU causes issues or isn't needed
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Uncomment if needed
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf # Use tf directly for optimizer
from tensorflow.keras.utils import Sequence
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
# Import ResNet50 instead of VGG16
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from PIL import Image
import warnings
import matplotlib.pyplot as plt # For plotting history later

# --- BoundingBoxDataGenerator Class (Using Cropping) ---
# This class remains the same as in the previous version
class BoundingBoxDataGenerator(Sequence):
    def __init__(self, dataframe, image_dir, batch_size, target_size, aug=None, shuffle=True):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.target_size = target_size # Expects (height, width) tuple
        self.aug = aug  # Keras ImageDataGenerator
        self.shuffle = shuffle

        # Get list of images and labels
        self.image_paths = dataframe['image'].values
        # Ensure labels are integers (0 or 1)
        self.labels = dataframe['label'].astype(np.int32).values
        self.bboxes = dataframe[['x1', 'y1', 'x2', 'y2']].values
        self.indices = np.arange(len(self.image_paths))
        self.on_epoch_end() # Shuffle indices if needed

    def __len__(self):
        # Calculate number of batches
        return int(np.floor(len(self.image_paths) / self.batch_size)) # Use floor for stability

    def __getitem__(self, index):
        # Generate indices of the batch
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        # Create batches using the calculated indices
        batch_images_paths = self.image_paths[batch_indices]
        batch_labels = self.labels[batch_indices]
        batch_bboxes = self.bboxes[batch_indices]

        # Load, crop, augment, and process images
        images = self.__load_crop_and_augment_images(batch_images_paths, batch_bboxes)

        # Ensure the images and labels are the correct types
        images = np.array(images, dtype=np.float32)
        # Labels should already be int32 from __init__

        # Handle case where image loading/cropping failed for all in batch
        if images.shape[0] == 0:
            # Return empty arrays with correct shapes if batch is empty
            return np.empty((0, *self.target_size, 3), dtype=np.float32), \
                   np.empty((0,), dtype=np.int32)

        return images, batch_labels

    def on_epoch_end(self):
        # Shuffle indices at the end of each epoch if shuffle is True
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __load_crop_and_augment_images(self, batch_images_paths, batch_bboxes):
        images = []
        for img_path, bbox in zip(batch_images_paths, batch_bboxes):
            # Load the image using PIL
            img_path_full = os.path.join(self.image_dir, img_path)
            try:
                img = Image.open(img_path_full).convert('RGB')
            except FileNotFoundError:
                warnings.warn(f"Image file not found: {img_path_full}. Skipping.")
                continue
            except Exception as e:
                 warnings.warn(f"Could not open image {img_path_full}: {e}. Skipping.")
                 continue

            # *** CROP the image using bounding box ***
            x1, y1, x2, y2 = map(int, bbox)

            # Basic check for valid box dimensions
            if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > img.width or y2 > img.height:
                warnings.warn(f"Invalid or out-of-bounds bbox {bbox} for image {img_path} with size {img.size}. Using full image.")
                pass # Keep original img, proceed to resize
            else:
                img = img.crop((x1, y1, x2, y2))

            # Resize the CROPPED (or fallback original) image
            img = img.resize(self.target_size) # Uses (width, height) order for PIL resize

            # Convert PIL image to NumPy array
            img_array = np.array(img, dtype=np.float32)

            # Apply augmentation and standardization using ImageDataGenerator
            if self.aug:
                img_array = self.aug.random_transform(img_array)
                img_array = self.aug.standardize(img_array) # Handles rescaling
            else:
                img_array = img_array / 255.0 # Fallback rescale if no aug

            images.append(img_array)

        return np.array(images, dtype=np.float32)

# --- Model Building Function (Using ResNet50) ---
# Note: Default dropout/l2 here are overridden by config variables below
def build_model(input_shape=(224, 224, 3), dropout_rate=0.15, l2_reg=0.0):
    # Load ResNet50 base, pre-trained on ImageNet, without the top classification layer
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # Initially, freeze the base model layers
    base_model.trainable = False

    # Build the sequential model
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        # Dense layer 1
        Dense(128, kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        Activation('relu'),
        Dropout(dropout_rate),
        # Dense layer 2
        Dense(64, kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        Activation('relu'),
        Dropout(dropout_rate / 2), # Adjusted second dropout based on the main rate
        # Output layer for binary classification
        Dense(1, activation='sigmoid')
    ], name="Audi_BMW_ResNet_Classifier") # Give the model a new name

    return model

# --- Configuration ---
# *** UPDATED INPUT_SHAPE for ResNet50 ***
INPUT_SHAPE = (224, 224, 3) # Optimal ResNet input size
BATCH_SIZE = 32 # Adjust if memory issues occur with larger images
EPOCHS = 100 # Max epochs, EarlyStopping will handle the actual number
LEARNING_RATE = 1e-3
# *** UPDATED Regularization based on previous discussion ***
DROPOUT_RATE = 0.15 # Lowered dropout
L2_REG = 0.0      # Removed L2 regularization initially
PATIENCE_EARLY_STOPPING = 10
PATIENCE_REDUCE_LR = 5
N_SPLITS = 5
TARGET_SPLIT = 0

# --- STEP 1: Load and Filter Data ---
print("--- Loading and Preparing Data ---")
try:
    df_all_train = pd.read_excel('stanford_cars/stanford_cars_with_class_names.xlsx', sheet_name='train')
    df_filtered = df_all_train[(df_all_train['class'] >= 12) & (df_all_train['class'] <= 38)].copy()
    df_filtered['label'] = df_filtered['class'].apply(lambda x: 0 if x <= 25 else 1) # 0=Audi, 1=BMW
    print(f"Loaded {len(df_all_train)} training samples.")
    print(f"Filtered to {len(df_filtered)} samples (Audi/BMW).")
except FileNotFoundError:
    print("ERROR: stanford_cars/stanford_cars_with_class_names.xlsx not found.")
    exit()
except Exception as e:
    print(f"ERROR: Could not load or process Excel file: {e}")
    exit()

# --- STEP 2: Stratified K-Fold Split ---
k_fold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
splits = list(k_fold.split(df_filtered['image'], df_filtered['label']))

if TARGET_SPLIT >= N_SPLITS:
    print(f"ERROR: TARGET_SPLIT ({TARGET_SPLIT}) is out of bounds for N_SPLITS ({N_SPLITS}).")
    exit()

train_idx, val_idx = splits[TARGET_SPLIT]
df_train = df_filtered.iloc[train_idx].copy()
df_val = df_filtered.iloc[val_idx].copy()

print(f"\nUsing Fold {TARGET_SPLIT + 1}/{N_SPLITS}:")
print(f"Training samples: {len(df_train)}")
print(f"Validation samples: {len(df_val)}")

# --- STEP 3: Data Augmentation ---
print("\n--- Setting up Data Augmentation ---")
# Using the slightly more aggressive augmentation from the fine-tuning script
# NOTE: ImageDataGenerator handles rescaling (1./255) within standardize
datagen_train = ImageDataGenerator(
    # rescale=1./255, # Rescaling is now handled by BoundingBoxDataGenerator standardize call
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest',
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input # Use ResNet50 preprocessing
)

# Only preprocessing for validation data
datagen_val = ImageDataGenerator(
    # rescale=1./255 # Handled by BoundingBoxDataGenerator standardize call
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input # Use ResNet50 preprocessing
)

# --- STEP 4: Create Data Generators ---
print("--- Creating Data Generators ---")
# Target size uses (height, width) which matches INPUT_SHAPE[:2]
train_gen = BoundingBoxDataGenerator(
    df_train,
    image_dir='stanford_cars/cars_train/',
    batch_size=BATCH_SIZE,
    target_size=INPUT_SHAPE[:2], # Uses (224, 224)
    aug=datagen_train,
    shuffle=True
)

val_gen = BoundingBoxDataGenerator(
    df_val,
    image_dir='stanford_cars/cars_train/',
    batch_size=BATCH_SIZE,
    target_size=INPUT_SHAPE[:2], # Uses (224, 224)
    aug=datagen_val,
    shuffle=False
)

# --- STEP 5: Build and Compile Model ---
print("--- Building Model with ResNet50 Base ---")
# Pass the config values explicitly
model = build_model(input_shape=INPUT_SHAPE, dropout_rate=DROPOUT_RATE, l2_reg=L2_REG)

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- STEP 6: Training (Frozen Base) ---
print("\n--- Starting Training (Frozen ResNet50 Base Model) ---")

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss',
                               patience=PATIENCE_EARLY_STOPPING,
                               restore_best_weights=True,
                               verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=PATIENCE_REDUCE_LR,
                              min_lr=1e-6,
                              verbose=1)
# Checkpoint to save the best model during training
checkpoint = ModelCheckpoint('audi_bmw_resnet_best_224.keras', # Updated checkpoint name
                             monitor='val_loss',
                             save_best_only=True,
                             mode='min',
                             verbose=1)

# Check if generators are empty before starting training
if len(train_gen) == 0 or len(val_gen) == 0:
    print("ERROR: Training or validation generator is empty. Check data paths and processing.")
else:
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[early_stopping, reduce_lr, checkpoint]
    )

    # --- STEP 7: Save Final Model and History ---
    print("\n--- Saving Final Model and History ---")

    # The best weights are already restored by EarlyStopping
    # Save the final model (which should have the best weights)
    model.save('audi_bmw_resnet_final_224.keras') # Updated save name
    print("Final ResNet model saved successfully to audi_bmw_resnet_final_224.keras")

    # Save history
    history_df = pd.DataFrame(history.history)
    history_csv_file = 'training_history_resnet_224.csv' # Updated history name
    try:
        history_df.to_csv(history_csv_file, index=False)
        print(f"Training history saved to {history_csv_file}")
    except Exception as e:
        print(f"Error saving history CSV: {e}")

    # --- STEP 8: Plot History ---
    print("\n--- Plotting Training History ---")

    # Check if history data exists
    if history and hasattr(history, 'history') and 'accuracy' in history.history and 'val_accuracy' in history.history:
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs_range = range(1, len(acc) + 1) # Correct epoch range

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy', marker='.', linestyle='-')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy', marker='.', linestyle='-')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy (ResNet 224x224)')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss', marker='.', linestyle='-')
        plt.plot(epochs_range, val_loss, label='Validation Loss', marker='.', linestyle='-')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss (ResNet 224x224)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)

        plt.suptitle('ResNet Model Training History (224x224)')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        # Save the plot before showing
        plot_filename = 'training_history_resnet_224_plot.png'
        plt.savefig(plot_filename)
        print(f"Training history plot saved to {plot_filename}")
        plt.show()
    else:
        print("Could not plot history: Missing history object or accuracy/validation accuracy data.")

print("\n--- Training Complete ---")