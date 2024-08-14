import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from efficientnet.tfkeras import EfficientNetB0
import matplotlib.pyplot as plt

print("Starting the script...")

# SETUP VARIABLES
input_size = 128
batch_size_num = 32
num_classes = 1  # Binary classification
dataset_path = './split_dataset/'

print(f"Input size: {input_size}, Batch size: {batch_size_num}, Num classes: {num_classes}")
print(f"Dataset path: {dataset_path}")

# TRAINING, VALIDATION, AND TEST PATHS
train_path = os.path.join(dataset_path, 'train')
val_path = os.path.join(dataset_path, 'val')
test_path = os.path.join(dataset_path, 'test')

print(f"Train path: {train_path}")
print(f"Validation path: {val_path}")
print(f"Test path: {test_path}")

# IMAGE DATA GENERATORS
print("Creating ImageDataGenerator for training data...")
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

print("Creating train_generator...")
train_generator = train_datagen.flow_from_directory(
    directory=train_path,
    target_size=(input_size, input_size),
    color_mode="rgb",
    class_mode="binary",
    batch_size=batch_size_num,
    shuffle=True
)
print(f"Train generator created. Number of batches: {len(train_generator)}")

print("Creating ImageDataGenerator for validation data...")
val_datagen = ImageDataGenerator(rescale=1/255)

print("Creating val_generator...")
val_generator = val_datagen.flow_from_directory(
    directory=val_path,
    target_size=(input_size, input_size),
    color_mode="rgb",
    class_mode="binary",
    batch_size=batch_size_num,
    shuffle=True
)
print(f"Validation generator created. Number of batches: {len(val_generator)}")

print("Creating ImageDataGenerator for test data...")
test_datagen = ImageDataGenerator(rescale=1/255)

print("Creating test_generator...")
test_generator = test_datagen.flow_from_directory(
    directory=test_path,
    target_size=(input_size, input_size),
    color_mode="rgb",
    class_mode=None,
    batch_size=1,
    shuffle=False
)
print(f"Test generator created. Number of batches: {len(test_generator)}")

# BUILD THE MODEL
print("Building the model...")
efficient_net = EfficientNetB0(
    weights='imagenet',
    input_shape=(input_size, input_size, 3),
    include_top=False,
    pooling='max'
)

print("Freezing EfficientNetB0 layers...")
efficient_net.trainable = False

model = Sequential()
model.add(efficient_net)
model.add(Dense(units=512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=num_classes, activation='sigmoid'))

print("Building the model...")
model.build((None, input_size, input_size, 3))

print("Compiling the model...")
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

print("Model summary:")
model.summary()

checkpoint_filepath = '.\\tmp_checkpoint'
print('Creating Directory: ' + checkpoint_filepath)
os.makedirs(checkpoint_filepath, exist_ok=True)

custom_callbacks = [
    EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=5,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=os.path.join(checkpoint_filepath, 'best_model.keras'),
        monitor='val_loss',
        mode='min',
        verbose=1,
        save_best_only=True
    )
]


# Train network
num_epochs = 20
history = model.fit(
    train_generator,                          
    epochs=num_epochs,
    steps_per_epoch=len(train_generator),
    validation_data=val_generator,
    validation_steps=len(val_generator),
    callbacks=custom_callbacks
)

print(history.history)

# Plot results
import numpy as np
import matplotlib.pyplot as plt
import os

# Assuming you have the history object from model.fit()
history = model.history.history

# Filter out the problematic data points
epochs = range(1, len(history['accuracy']) + 1)
acc = [a for a in history['accuracy'] if a != 0]
val_acc = [a for a in history['val_accuracy'] if a != 0]
loss = [l for l in history['loss'] if l != 0]
val_loss = [l for l in history['val_loss'] if l != 0]

# Adjust epochs to match the filtered data
epochs = range(1, len(acc) + 1)

# PLOT TRAINING AND VALIDATION ACCURACY
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'bo-', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b*-', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# PLOT TRAINING AND VALIDATION LOSS
plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'ro-', label='Training Loss')
plt.plot(epochs, val_loss, 'r*-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()

# SAVE THE PLOTS
checkpoint_filepath = 'results//'

combined_plot_path = os.path.join(checkpoint_filepath, 'training_validation_combined.png')

print("saved at path: " , combined_plot_path)

plt.savefig(combined_plot_path)
plt.show()

# Print some statistics
print(f"Number of valid epochs: {len(acc)}")
print(f"Final training accuracy: {acc[-1]:.4f}")
print(f"Final validation accuracy: {val_acc[-1]:.4f}")
print(f"Final training loss: {loss[-1]:.4f}")
print(f"Final validation loss: {val_loss[-1]:.4f}")


# SAVE THE FINAL BEST MODEL
final_model_path = os.path.join(checkpoint_filepath, 'best_model.keras')
model.save(final_model_path)

# load the saved model that is considered the best
best_model = load_model(os.path.join(checkpoint_filepath, 'best_model.keras'))

# Generate predictions
test_generator.reset()

preds = best_model.predict(
    test_generator,
    verbose = 1
)

test_results = pd.DataFrame({
    "Filename": test_generator.filenames,
    "Prediction": preds.flatten()
})
print(test_results)
