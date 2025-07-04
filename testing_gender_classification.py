# Importing Libraries and Modules
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path

# Set image size and model path
IMG_SIZE = (224, 224)
MODEL_PATH = "gender_classification.h5"

# Load the trained model
model = load_model(MODEL_PATH)

#Function to load and resize Images
def load_data(file_path):
    images = []

    for img in sorted(os.listdir(file_path)):
        img_path = os.path.join(file_path, img)
        image = Image.open(img_path).convert("RGB") # Converting images to RGB format
        image = image.resize(IMG_SIZE) # Resizing Images
        images.append(image)

    return images

# Taking Path as Input
path_val = input("Enter validation Path:")
path_val = Path(path_val)

# Splitting Male and Female paths
val_file_path_male = os.path.join(path_val, "male")
val_file_path_female = os.path.join(path_val, "female")

# Loading Male and Female Images Separately
test_male_set = load_data(val_file_path_male)
test_female_set = load_data(val_file_path_female)

# Making a list of Male and Female Labels
test_male_label = [1] * len(test_male_set)
test_female_label = [0] * len(test_female_set)

# Converting labels and set into numpy array
test_male_label = np.array(test_male_label)
test_female_label = np.array(test_female_label)
test_male_set = np.stack(test_male_set)
test_female_set = np.stack(test_female_set)

# Combining Male + Female data
test_set = np.concatenate((test_male_set, test_female_set))
test_label =  np.concatenate((test_male_label, test_female_label))

# Normalizing Images
test_set = test_set/255

# Making prediction on test set using loaded model
preds = model.predict(test_set)
preds_binary = (preds > 0.5).astype(int).flatten()

# Calculating Accuracy, Precision, Recall and F1 score
accuracy = accuracy_score(test_label, preds_binary)
precision = precision_score(test_label, preds_binary)
recall = recall_score(test_label, preds_binary)
f1 = f1_score(test_label, preds_binary)

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
