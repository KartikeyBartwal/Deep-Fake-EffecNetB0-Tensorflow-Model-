import os
import pandas as pd
import logging

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from efficientnet.tfkeras import EfficientNetB0
import matplotlib.pyplot as plt

# SETUP LOGGING
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

os.chdir('..')

logging.info("Starting the script...")

# SETUP VARIABLES
input_size = 128

batch_size_num = 32

num_classes = 1  # Binary classification

dataset_path = './split_dataset/'

logging.info(f"Input size: {input_size}, Batch size: {batch_size_num}, Num classes: {num_classes}")

logging.info(f"Dataset path: {dataset_path}")

# TRAINING, VALIDATION, AND TEST PATHS
train_path = os.path.join(dataset_path, 'train')

val_path = os.path.join(dataset_path, 'val')

test_path = os.path.join(dataset_path, 'test')

logging.info(f"Train path: {train_path}")

logging.info(f"Validation path: {val_path}")

logging.info(f"Test path: {test_path}")

# IMAGE DATA GENERATORS
logging.info("Creating ImageDataGenerator for training data...")

train_datagen = ImageDataGenerator(
    rescale=1/255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

logging.info("Creating train_generator...")

train_generator = train_datagen.flow_from_directory(
    directory=train_path,
    target_size=(input_size, input_size),
    color_mode="rgb",
    class_mode="binary",
    batch_size=batch_size_num,
    shuffle=True
)

logging.info(f"Train generator created. Number of batches: {len(train_generator)}")

logging.info("Creating ImageDataGenerator for validation data...")

val_datagen = ImageDataGenerator(rescale=1/255)

logging.info("Creating val_generator...")

val_generator = val_datagen.flow_from_directory(

    directory=val_path,
    target_size=(input_size, input_size),
    color_mode="rgb",
    class_mode="binary",
    batch_size=batch_size_num,
    shuffle=True

)

logging.info(f"Validation generator created. Number of batches: {len(val_generator)}")

logging.info("Creating ImageDataGenerator for test data...")

test_datagen = ImageDataGenerator(rescale=1/255)

logging.info("Creating test_generator...")

test_generator = test_datagen.flow_from_directory(
   
    directory=test_path,
    target_size=(input_size, input_size),
    color_mode="rgb",
    class_mode=None,
    batch_size=1,
    shuffle=False

)

logging.info(f"Test generator created. Number of batches: {len(test_generator)}")

# BUILD THE MODEL
logging.info("Building the model...")

efficient_net = EfficientNetB0(

    weights='imagenet',
    input_shape=(input_size, input_size, 3),
    include_top=False,
    pooling='max'

)

logging.info("Freezing EfficientNetB0 layers...")

efficient_net.trainable = False

model = Sequential()

model.add(efficient_net)

model.add(Dense(units=512, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(units=128, activation='relu'))

model.add(Dense(units=num_classes, activation='sigmoid'))


logging.info("Building the model...")

model.build((None, input_size, input_size, 3))

logging.info("Compiling the model...")

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

logging.info("Model summary:")
model.summary()

# CREATE CHECKPOINT DIRECTORY
checkpoint_filepath = './tmp_checkpoint'

logging.info(f'Creating Directory: {checkpoint_filepath}')

os.makedirs(checkpoint_filepath, exist_ok=True)

# SET CUSTOM CALLBACKS
custom_callbacks = [
    None 
]

# TRAIN NETWORK
num_epochs = 20
logging.info(f"Starting model training for {num_epochs} epochs...")

history = model.fit(

    train_generator,
    epochs=num_epochs,
    steps_per_epoch=len(train_generator),
    # validation_data=val_generator,
    # validation_steps=len(val_generator),
    batch_size=batch_size_num,

)

logging.info("Training completed. Printing training history...")

logging.info(history.history)

# PLOT RESULTS
logging.info("Plotting results...")

acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
loss = history.history['loss']
# val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# PLOT TRAINING AND VALIDATION ACCURACY
plt.figure(figsize=(10, 5))
plt.plot(epochs, acc, 'bo-', label='Training Accuracy')
# plt.plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
plt.title('Training Accuracy')
plt.legend()
plt.savefig('tmp_checkpoint/training_validation_accuracy.png')
logging.info("Accuracy plot saved.")

plt.figure()

# PLOT TRAINING AND VALIDATION LOSS
plt.figure(figsize=(10, 5))

plt.plot(epochs, loss, 'bo-', label='Training Loss')
# plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')

plt.title('Training and Validation Loss')


plt.legend()

plt.savefig('tmp_checkpoint/training_validation_loss.png')

logging.info("Loss plot saved.")

# SAVE THE MODEL 
logging.info("Saving the model...")
model.save("tmp_checkpoint/best_model.h5")

logging.info("Model saved successfully.")

logging.info("Script execution completed.")
