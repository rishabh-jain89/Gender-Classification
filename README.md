# Gender-Classification
This project implements a **Convolutional Neural Network (CNN)** to classify facial images as **male** or **female** using TensorFlow and Keras. The model is trained on a structured dataset with image augmentation, regularization, and performance evaluation.


---

## ⚙️ Requirements

Install dependencies:

```bash
pip install tensorflow numpy pandas scikit-learn pillow matplotlib
```

## 🚀 How to Run
✅ Train the Model
```bash
python training_gender_classification.py
```

- Loads and augments data

- Trains a CNN from scratch

- Saves the model as gender_recognition.h5

✅ Evaluate the Model
```bash
python testing_gender_classification.py
```
- Takes the validation folder path as input 

- Loads validation set

- Loads the saved model

- Computes accuracy, precision, recall, F1 score

## 🧠 Model Details
- Architecture: CNN with Conv2D, MaxPooling, BatchNormalization, Dropout

- Loss Function: Binary Crossentropy with label smoothing

- Optimizer: Adam

- Data Augmentation: Rotation, shifting, zoom, brightness adjustment, channel shifting

- Class Balancing: Synthetic augmentation of the underrepresented class (female)

