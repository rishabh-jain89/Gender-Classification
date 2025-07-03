import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,  MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

train_file_path_male = "/home/rishabh-jain/Desktop/Syscom/New Dataset/Comys_Hackathon5/Task_A/train/male"
train_file_path_female = "/home/rishabh-jain/Desktop/Syscom/New Dataset/Comys_Hackathon5/Task_A/train/female"

val_file_path_male = "/home/rishabh-jain/Desktop/Syscom/New Dataset/Comys_Hackathon5/Task_A/val/male"
val_file_path_female = "/home/rishabh-jain/Desktop/Syscom/New Dataset/Comys_Hackathon5/Task_A/val/female"

img_size = (224, 224)

def load_data(file_path):
    images = []
    name = []

    for img in sorted(os.listdir(file_path)):
        img_path = os.path.join(file_path, img)
        image = Image.open(img_path).convert("RGB")
        image = image.resize(img_size)
        images.append(image)
        name.append(img)

    return images, name

train_male_set, train_male_name = load_data(train_file_path_male)
train_male_label = [1] * len(train_male_set)
train_male_label = np.array(train_male_label)
train_female_set, train_female_name = load_data(train_file_path_female)
train_female_label = [0] * len(train_female_set)
train_female_label = np.array(train_female_label)

test_male_set, test_male_name = load_data(val_file_path_male)
test_male_label = [1] * len(test_male_set)
test_male_label = np.array(test_male_label)
test_female_set, test_female_name = load_data(val_file_path_female)
test_female_label = [0] * len(test_female_set)
test_female_label = np.array(test_female_label)

print(train_male_label.shape, test_male_label.shape)
print(train_female_label.shape, test_female_label.shape)

datagen = ImageDataGenerator(
    rotation_range=45, 
    width_shift_range=0.25,
    height_shift_range=0.25,
    shear_range=0.25,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,  
    brightness_range=[0.7, 1.4], 
    channel_shift_range=50,  
    fill_mode='reflect'  
)

female_imgs_aug = []
female_labels_aug = []

for img in train_female_set:  
    img = np.expand_dims(img, axis=0)
    aug_iter = datagen.flow(img, batch_size=1)
    for i in range(3):  
        aug_img = next(aug_iter)[0]
        female_imgs_aug.append(aug_img)
        female_labels_aug.append(0)

female_imgs_aug = np.stack(female_imgs_aug)
female_labels_aug = np.array(female_labels_aug)

train_female_set = np.stack(train_female_set)
train_female_aug_set = np.concatenate((train_female_set, female_imgs_aug))

train_female_label = np.array(train_female_label)
train_female_aug_label = np.concatenate((train_female_label, female_labels_aug))

print(train_female_aug_set.shape, train_female_aug_label.shape)

train_female_aug_set = np.clip(train_female_aug_set, 0, 255).astype(np.uint8)

i = random.randint(0, train_female_aug_set.shape[0])

plt.imshow(train_female_aug_set[i])
plt.axis('off')  # Hide axes
plt.title(f"Label: {train_female_aug_label[i]} ")
plt.show()

j = random.randint(0, train_female_set.shape[0])

plt.imshow(train_female_set[j])
plt.axis('off')  # Hide axes
plt.title(f"Label: {train_female_label[j]} ")
plt.show()

def stack_images(images, labels):
    fig, axes = plt.subplots(4, 5, figsize=(15, 12)) 

    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i])
            ax.set_title(str(labels[i]))   
            ax.axis('off')                 
        else:
            ax.axis('off')                 

    plt.tight_layout()
    plt.show()

train_female_aug_set = np.stack(train_female_aug_set)

train_male_set = np.stack(train_male_set)
train_set = np.concatenate((train_male_set, train_female_aug_set))

train_label = np.concatenate((train_male_label, train_female_aug_label))

test_male_set = np.stack(test_male_set)
test_female_set = np.stack(test_female_set)
test_set = np.concatenate((test_male_set, test_female_set))
test_label =  np.concatenate((test_male_label, test_female_label))

print(train_set.shape, train_label.shape)
print(test_set.shape, test_label.shape)

combined = list(zip(train_set, train_label))
random.seed(42)
random.shuffle(combined)
train_set, train_label = zip(*combined)

combined = list(zip(test_set, test_label))
random.shuffle(combined)
test_set, test_label = zip(*combined)

train_set = np.stack(train_set)
train_label = np.stack(train_label)

test_set = np.stack(test_set)
test_label = np.stack(test_label)

print(type(train_set), type(test_set))
print(type(train_label), type(test_label))
print(train_set.shape, test_set.shape)
print(train_label.shape, test_label.shape)

stack_images(train_set, train_label)

weights = compute_class_weight('balanced', classes=np.unique(train_label), y=train_label)
class_weights = {0: weights[0], 1: weights[1]}

train_set = (train_set )/255
test_set = (test_set )/255

unique, counts = np.unique(train_label, return_counts=True)
print("Train label distribution:", dict(zip(unique, counts)))

unique, counts = np.unique(test_label, return_counts=True)
print("Test label distribution:", dict(zip(unique, counts)))

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

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

model.compile(loss = tf.keras.losses.BinaryCrossentropy(label_smoothing = 0.1), optimizer = 'adam', metrics = ['accuracy'])

model.fit(train_set, train_label, epochs = 20, batch_size = 64)

model.evaluate(test_set, test_label, return_dict = True, batch_size = 64)

# Predict on full test set
preds = model.predict(test_set)
preds_binary = (preds > 0.5).astype(int).flatten()

accuracy = accuracy_score(test_label, preds_binary)
precision = precision_score(test_label, preds_binary)
recall = recall_score(test_label, preds_binary)
f1 = f1_score(test_label, preds_binary)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
cm = confusion_matrix(test_label, preds_binary)
print(cm)

model.save("gender_recognition.h5")