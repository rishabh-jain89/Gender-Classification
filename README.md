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

## 🎯 Evaluation Metrics (on train data)

| Metric     | Value     |
|------------|-----------|
| Accuracy   | `0.9916`  |
| Precision  | `0.9846`  |
| Recall     | `0.9987`  |
| F1-Score   | `0.9916`  |

## 🎯 Evaluation Metrics (on validation)

| Metric     | Value     |
|------------|-----------|
| Accuracy   | `0.9100`  |
| Precision  | `0.9043`  |
| Recall     | `0.9842`  |
| F1-Score   | `0.9426`  |

## 🧠 Model Details
- Architecture: CNN with Conv2D, MaxPooling, BatchNormalization, Dropout

- Loss Function: Binary Crossentropy with label smoothing

- Optimizer: Adam

- Data Augmentation: Rotation, shifting, zoom, brightness adjustment, channel shifting

- Class Balancing: Synthetic augmentation of the underrepresented class (female)

## 🤝 Contributors

**Prakshay Saini**  
B.Tech CSE, IIIT Guwahati  
prakshay.saini23b@iiitg.ac.in

**Rishab Jain**  
B.Tech CSE, IIIT Guwahati  
rishab.jain23b@iiitg.ac.in
