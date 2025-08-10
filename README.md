#  Deep Learning Project – Image Classification 

*COMPANY* :  CODETECH IT SOLUTIONS

*NAME* :  S.NIRANJANA

*INTERN ID* :  CT04DH2216

*DOMAIN* :  DATA SCIENCE

*DURATION* :  4WEEKS

*MENTOR* :  NEELA SANTOSH KUMAR

---
## 🔍 Overview

This repository contains a **Deep Learning Image Classification project** implemented using **TensorFlow** and **Keras**. It was developed as part of the **CodTech Internship**, where the goal was to build, train, and evaluate a neural network model on a real-world image dataset.

The project includes:
- A functional Convolutional Neural Network (CNN)
- Training and validation accuracy/loss plots
- Predictions on test images with visual outputs
- Clear, reproducible code in Jupyter Notebook format

---

## 🎯 Project Objective

The main goal is to demonstrate knowledge of deep learning by:
- Building a model capable of distinguishing between image classes
- Training it using a well-known dataset (e.g. CIFAR-10 or Fashion-MNIST)
- Visualizing the performance and interpreting results

This project lays the groundwork for future research and development in computer vision and AI.

---

## 🧰 Tools & Technologies

- **Python**
- **TensorFlow / Keras**
- **NumPy**, **Matplotlib**, **Seaborn**
- **Jupyter Notebook**

---

## 📊 Dataset

We used the **CIFAR-10** dataset, which consists of 60,000 32x32 color images in 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck), with 6,000 images per class.

- Training set: 50,000 images
- Test set: 10,000 images

The dataset is directly loaded using TensorFlow datasets or Keras API.

---

## 🧠 Model Architecture

A basic Convolutional Neural Network (CNN) with the following layers:

- Conv2D → ReLU → MaxPooling
- Conv2D → ReLU → MaxPooling
- Flatten
- Dense → Dropout
- Output (Softmax for multiclass classification)

---

## 📈 Results & Visualizations

- Training & validation accuracy/loss graphs  
- Sample image predictions with labels  
- Confusion matrix (optional)  

### 📸 Example Visualization

```python
import matplotlib.pyplot as plt

# Plotting accuracy
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
