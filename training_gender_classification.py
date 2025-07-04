#Importing Libraries and Modules
import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,  MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import CSVLogger
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#Defining dataset Paths
train_path = "/home/rishabh-jain/Desktop/Syscom/New Dataset/Comys_Hackathon5/Task_A/train"
train_file_path_male = os.path.join(train_path, "male")
train_file_path_female = os.path.join(train_path, "female")

#Setting Image size and seed
img_size = (224, 224)
random.seed(42)

#Function to load and resize Images
def load_data(file_path):
    images = []

    for img in sorted(os.listdir(file_path)):
        img_path = os.path.join(file_path, img)
        image = Image.open(img_path).convert("RGB") #converting images to RGB format
        image = image.resize(img_size) #Resizing Images
        images.append(image)
    return images

 # Loading Male and Female Images Separately
train_male_set= load_data(train_file_path_male)
train_female_set = load_data(train_file_path_female)

# Making a list of Male and Female Labels
train_male_label = [1] * len(train_male_set) 
train_female_label = [0] * len(train_female_set)

#Converting labels into numpy array
train_male_label = np.array(train_male_label)
train_female_label = np.array(train_female_label)

#Defining Data Augmentation Pipeline
datagen = ImageDataGenerator(
    rotation_range=45, # Randomly rotate images by upto 45 degrees
    width_shift_range=0.25, # Shift image horizontally by up to 25% of the width
    height_shift_range=0.25, # Shift image vertically by up to 25% of the height
    shear_range=0.25, #  Apply shearing (tilting) transformations up to 25 degrees
    zoom_range=0.3, #  Randomly zoom in/out by up to 30%
    horizontal_flip=True,#   Randomly flip images horizontally (mirror image)
    vertical_flip=True,  #  Randomly flip images vertically (upside down)
    brightness_range=[0.7, 1.4], #  Adjust brightness randomly between 70% (darker) and 140% (brighter)
    channel_shift_range=50,  #  Randomly shift color channels by up to 50 units (simulate lighting changes)
    fill_mode='reflect'  #  Filling strategy for newly created pixels (reflects pixel values at the border)
)

# Augment female images to balance dataset
female_imgs_aug = []
female_labels_aug = []

for img in train_female_set:  
    img = np.expand_dims(img, axis=0) # expand batch dimensions
    aug_iter = datagen.flow(img, batch_size=1)
    for i in range(3):  # Generate 3 augmentations per image
        aug_img = next(aug_iter)[0]
        female_imgs_aug.append(aug_img)
        female_labels_aug.append(0)

# Stack Augmented data into arrays
female_imgs_aug = np.stack(female_imgs_aug)
female_labels_aug = np.array(female_labels_aug)

# Combining Original + Augmented female data
train_female_set = np.stack(train_female_set)
train_female_aug_set = np.concatenate((train_female_set, female_imgs_aug))
train_female_aug_label = np.concatenate((train_female_label, female_labels_aug))

# Clip pizel values to vaid range and convert to uint8
train_female_aug_set = np.clip(train_female_aug_set, 0, 255).astype(np.uint8)

# stack final female data
train_female_aug_set = np.stack(train_female_aug_set)

# Combining Male and Female data
train_male_set = np.stack(train_male_set)
train_set = np.concatenate((train_male_set, train_female_aug_set))
train_label = np.concatenate((train_male_label, train_female_aug_label))

# Shuffling the dataset
combined = list(zip(train_set, train_label))
random.shuffle(combined)
train_set, train_label = zip(*combined)
train_set = np.stack(train_set)
train_label = np.stack(train_label)

# Normalizing Images
train_set = (train_set )/255

#Checking for GPU Availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Defining CNN Model Architecture
inp_shape = (224, 224, 3)
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu',kernel_regularizer=l2(0.001), input_shape=inp_shape))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu',kernel_regularizer=l2(0.001), kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001), kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2))) 

model.add(Conv2D(256, (3, 3), activation='relu',kernel_regularizer=l2(0.001),  kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001))) 
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))  

# Compiling the Model
model.compile(loss = tf.keras.losses.BinaryCrossentropy(label_smoothing = 0.1), optimizer = 'adam', metrics = ['accuracy'])

# Setting up logger before training
csv_logger = CSVLogger('training_results_gender_classification.txt', append=False)


# Training the Model
model.fit(train_set, train_label, epochs = 30, batch_size = 64, callbacks = [csv_logger])

# Save the Trained Model
model.save("gender_classification.h5")

# Making prediction on test set using loaded model
preds = model.predict(train_set)
preds_binary = (preds > 0.5).astype(int).flatten()

# Calculating Accuracy, Precision, Recall and F1 score
accuracy = accuracy_score(train_label, preds_binary)
precision = precision_score(train_label, preds_binary)
recall = recall_score(train_label, preds_binary)
f1 = f1_score(train_label, preds_binary)

# Creating a log string
log = (
    f"Accuracy:  {accuracy:.4f}\n"
    f"Precision: {precision:.4f}\n"
    f"Recall:    {recall:.4f}\n"
    f"F1 Score:  {f1:.4f}\n"
)

# Write to file
with open("validation_results_gender_classification.txt", "a") as f:
    f.write(log)