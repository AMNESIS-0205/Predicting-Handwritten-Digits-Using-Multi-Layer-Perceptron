# 🔢 Predicting Handwritten Digits Using Multi-Layer Perceptron (MLP)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://predicting-handwritten-digits-using-multi-layer-perceptron-c7a.streamlit.app/)

## 🚀 Live Demo
**[Launch the Interactive App Here](https://predicting-handwritten-digits-using-multi-layer-perceptron-c7a.streamlit.app/)**

## 📌 Project Overview
This project focuses on building a neural network model using a **Multi-Layer Perceptron (MLP)** to classify handwritten digits from the **MNIST dataset**. The model learns complex non-linear patterns from raw pixel data to recognize digits ranging from 0 to 9. This project serves as a foundational step in understanding deep learning fundamentals before moving to advanced architectures like CNNs.

## ❓ Problem Statement
Digit recognition is a fundamental problem in computer vision with critical applications in postal automation and banking. While traditional machine learning algorithms struggle with large feature spaces like image pixels, this MLP model effectively learns non-linear relationships between input features and target classes.

## ✨ Key Features
* **Interactive Sketchpad**: Draw digits 0-9 directly in the UI for real-time recognition.
* **Confidence Dashboard**: Visualizes the model's "thinking process" through probability bar charts.
* **Professional Sidebar**: Detailed project breakdown and technical metrics.
* **Optimized Performance**: Uses a trained "brain" to deliver instant predictions.

## 🏗️ Model Architecture
The MLP architecture consists of an input layer, two hidden layers, and an output layer:
* **Input Layer**: 784 neurons (representing flattened 28x28 grayscale pixels).
* **Hidden Layers**: Two dense layers with **ReLU** activation functions to capture non-linearities.
* **Output Layer**: 10 neurons with **Softmax** activation to provide probability scores for each class.
* **Optimization**: Trained using **Cross-Entropy Loss** and the **Adam Optimizer** for faster convergence.

## 📊 Methodology
1.  **Preprocessing**: Loaded and normalized MNIST dataset images to improve convergence.
2.  **Model Design**: Built an MLP with specific activations to map pixels to digits.
3.  **Training**: Optimized weights using backpropagation to minimize error rates.
4.  **Evaluation**: Validated performance using accuracy metrics and a confusion matrix.
5.  **Comparison**: Benchmarked against simple models like Logistic Regression to verify model efficacy.

## 🛠️ Tech Stack
* **Deep Learning**: TensorFlow / Keras
* **Web Framework**: Streamlit
* **Mathematics & Vision**: NumPy, OpenCV
* **Environment**: Google Colab (Training) and VS Code (Deployment)

## 💻 How to Run Locally
1.  Clone this repository:
    ```bash
    git clone [https://github.com/AMNESIS-0205/Predicting-Handwritten-Digits-Using-Multi-Layer-Perceptron.git](https://github.com/AMNESIS-0205/Predicting-Handwritten-Digits-Using-Multi-Layer-Perceptron.git)
    ```
2.  Install required packages:
    ```bash
    pip install -r requirements.txt
    ```
3.  Launch the application:
    ```bash
    streamlit run app.py
    ```

---
**Developed by Arpit Malviya** *Project
