import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from pathlib import Path

# Set image size and model path
IMG_SIZE = (224, 224)
MODEL_PATH = "gender_recognition.h5"

# Load the trained model
model = load_model(MODEL_PATH)

def load_data(file_path):
    images = []
    name = []

    for img in sorted(os.listdir(file_path)):
        img_path = os.path.join(file_path, img)
        image = Image.open(img_path).convert("RGB")
        image = image.resize(IMG_SIZE)
        images.append(image)
        name.append(img)

    return images, name

path_val = input("Enter validation Path:")
path_val = Path(path_val)

val_file_path_male = os.path.join(path_val, "male")
val_file_path_female = os.path.join(path_val, "female")


test_male_set, test_male_name = load_data(val_file_path_male)
test_male_label = [1] * len(test_male_set)
test_male_label = np.array(test_male_label)
test_female_set, test_female_name = load_data(val_file_path_female)
test_female_label = [0] * len(test_female_set)
test_female_label = np.array(test_female_label)

test_male_set = np.stack(test_male_set)
test_female_set = np.stack(test_female_set)
test_set = np.concatenate((test_male_set, test_female_set))
test_label =  np.concatenate((test_male_label, test_female_label))


test_set = test_set/255

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