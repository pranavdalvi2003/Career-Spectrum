# # **CareerTrackPredict – AI-Powered Career Recommendation System**

CareerTrackPredict is a machine-learning–based career prediction system developed during my internship at **CareerSpectrum** as a **Data Science Intern**.
The project focuses on predicting the most suitable career paths for students based on their skills, interests, academic performance, and psychometric parameters.

This system uses both **mathematical modeling** and **neural network–based deep learning** to generate accurate and meaningful predictions, helping students make informed decisions about their career paths.

---

## **Table of Contents**

1. [Overview](#overview)
2. [Features](#features)
3. [Architecture Overview](#architecture-overview)
4. [Tech Stack](#tech-stack)
5. [Dataset](#dataset)
6. [Modeling Approach](#modeling-approach)

   * Mathematical Model
   * Neural Network Model
7. [Installation](#installation)
8. [Results](#results)
9. [Future Enhancements](#future-enhancements)

---

# ## **Overview**

CareerTrackPredict is an AI-based recommendation system that learns patterns from student profiles and predicts the top career fields suitable for them.
The model processes:

* Academic scores
* Skills & interests
* Behavioral insights
* Psychometric parameters

The project aimed at improving career guidance services by integrating **machine learning**, **deep learning**, and **data preprocessing pipelines** into a unified predictive workflow.

---

# ## **Features**

### 📊 **Data Preprocessing & Analysis**

* Cleaning and normalization
* Feature extraction
* Statistical analysis
* Outlier detection & handling

### 🧮 **Mathematical Model**

* Baseline model for structured prediction
* Probability estimation using mathematical functions
* Feature weighting and scoring mechanisms

### 🤖 **Neural Network Model**

* Custom-built multi-layer neural network
* SoftMax activation for multi-class prediction
* Backpropagation & parameter optimization
* High accuracy on test dataset

### 🔍 **Career Recommendation**

* Predicts top 3 most suitable career paths
* Generates confidence scores
* Helps students choose ideal education streams

### 📈 **Performance Evaluation**

* Confusion matrix
* Accuracy, precision, recall
* Loss/accuracy curves

---

# ## **Architecture Overview**

```ascii
                      ┌────────────────────────┐
                      │     Student Dataset     │
                      │ (scores, interests etc)│
                      └──────────────┬─────────┘
                                     │
                                     ▼
                  ┌─────────────────────────────────┐
                  │     Data Preprocessing Layer     │
                  │ - Cleaning & Normalization       │
                  │ - Feature Engineering            │
                  └─────────────────┬───────────────┘
                                    │
               ┌────────────────────┴────────────────────┐
               │                                           │
               ▼                                           ▼
   ┌─────────────────────┐                      ┌─────────────────────────┐
   │ Mathematical Model   │                      │ Neural Network Model    │
   │ (Baseline Predictor) │                      │ (Deep Learning)         │
   └──────────┬──────────┘                      └──────────┬──────────────┘
              │                                            │
              └───────────────────────┬────────────────────┘
                                      ▼
                           ┌───────────────────────┐
                           │  Career Predictions   │
                           │  Top Fields + Scores │
                           └───────────────────────┘
```

---

# ## **Tech Stack**

### **Languages**

* Python
* Jupyter Notebook

### **Libraries**

* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* TensorFlow / Keras

### **Tools**

* Google Colab
* CareerSpectrum data portal
* Git

---

# ## **Dataset**

The dataset consisted of anonymized student information, including:

* Academic subjects & scores
* Interests & skills
* Cognitive abilities
* Behavioral traits
* Past career inclination data

The data was preprocessed, analyzed, and structured before training the models.

---

# ## **Modeling Approach**

## **1. Mathematical Model**

The mathematical model served as the project’s baseline. It involved:

* Weight assignment to student attributes
* Score aggregation
* Normalized probability computation
* Linear/non-linear transformation of features

This model provided interpretable predictions and acted as a benchmark for neural network performance.

---

## **2. Neural Network Model**

The core model was a **multi-layer neural network** designed for multi-class classification.

### **Key Highlights**

* Input layer: Student feature vector
* Hidden layers: Dense layers with ReLU
* Output layer: SoftMax activation
* Loss function: Categorical cross-entropy
* Optimizer: Adam
* Training: Backpropagation with mini-batch gradient descent

### **Outcomes**

* Improved accuracy compared to baseline
* Smooth convergence
* Robust performance across test samples

---

# ## **Installation**

Clone repository (if applicable):

```bash
git clone https://github.com/pranavdalvi2003/CareerSpectrum.git
cd CareerSpectrum
```

Install required libraries:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install numpy pandas scikit-learn matplotlib tensorflow
```

---

### Steps:

1. Upload/Load Dataset
2. Preprocess data
3. Train mathematical model
4. Train neural network model
5. Generate predictions
6. Visualize performance metrics

---

# ## **Results**

### ✔ High prediction accuracy with neural network

### ✔ Smooth training convergence

### ✔ Clear separation between career categories

### ✔ Useful insights for students & counselors

The model successfully predicted suitable careers such as:

* Engineering
* Management
* Arts
* Medical
* Commerce
* IT / Technical roles

---

# ## **Future Enhancements**

* Add deep learning–based psychometric analysis
* Integrate NLP-based resume parsing
* Develop a live web app for student interaction
* Add explainability using SHAP/LIME
* Improve dataset diversity
* Deploy model as an API
