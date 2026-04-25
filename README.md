🏥 Diabetes Prediction System
AI-Driven Healthcare Analytics — Machine Learning, Deep Learning & Clustering

Certificate in Artificial Intelligence — Take-Home Assignment
Student No: `IT/CAI/26/03/0031` | Domain: Healthcare Analytics

📌 Overview

This project builds an end-to-end "Diabetes Prediction System" using the Pima Indians Diabetes Dataset. It applies three major AI methodologies to classify patients as diabetic or non-diabetic and to discover hidden patient risk groups — without requiring specialist clinical input.

| Component | Approach | Best Result |
|-----------|----------|-------------|
| 🌲 Machine Learning | Random Forest, Logistic Regression, SVM | ~85–90% accuracy |
| 🧠 Deep Learning | Multi-Layer Perceptron (MLP) | ~78–80% accuracy |
| 🔵 Clustering | K-Means + Hierarchical (K=3) | 3 risk groups identified |

📂 Table of Contents

- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Methodology](#-methodology)
  - [Data Preprocessing](#data-preprocessing)
  - [Machine Learning](#machine-learning--part-a)
  - [Deep Learning](#deep-learning--part-b)
  - [Clustering](#clustering--part-c)
- [Results](#-results)
- [Visualisations](#-visualisations)
- [Key Findings](#-key-findings)
- [Future Work](#-future-work)
- [References](#-references)


 📊 Dataset

- **Source:** [Pima Indians Diabetes Database — Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Records:** 768 patient observations
- **Features:** 8 clinical input features + 1 binary target
- **Classes:** 0 = Non-diabetic, 1 = Diabetic
- **Class balance:** ~65% non-diabetic / ~35% diabetic

| Feature | Unit | Description |
|---------|------|-------------|
| `Pregnancies` | Count | Number of pregnancies |
| `Glucose` | mg/dL | Plasma glucose concentration (2-hr oral glucose tolerance test) |
| `BloodPressure` | mm Hg | Diastolic blood pressure |
| `SkinThickness` | mm | Triceps skin fold thickness |
| `Insulin` | mu U/ml | 2-hour serum insulin level |
| `BMI` | kg/m² | Body mass index |
| `DiabetesPedigreeFunction` | Score | Hereditary diabetes risk score |
| `Age` | Years | Patient age |
| `Outcome` | Binary | **Target:** 0 = Non-diabetic, 1 = Diabetic |

📁 Project Structure

```
diabetes-prediction-system/
│
├── data/
│   └── diabetes.csv                         # Pima Indians Diabetes Dataset
│
├── notebooks/
│   ├── 01_EDA_Preprocessing.ipynb           # Exploratory data analysis
│   ├── 02_Machine_Learning.ipynb            # ML models (LR, RF, SVM)
│   ├── 03_Deep_Learning.ipynb               # MLP neural network
│   └── 04_Clustering.ipynb                  # K-Means & Hierarchical clustering
│
├── images/
│   ├── Feature_Correlation_Heatmap.jpeg
│   ├── Logistic_Regression_Confusion_Matrix.jpeg
│   ├── Random_Forest_Confusion_Matrix.jpeg
│   ├── Tuned_Random_Forest_Confusion_Matrix.jpeg
│   ├── SVM_Confusion_Matrix.jpeg
│   ├── Deep_Learning_Accuracy_Curve.jpeg
│   └── Patient_Clusters_Glucose_vs_BMI.jpeg
│
├── models/
│   ├── random_forest_model.pkl              # Saved best ML model
│   └── mlp_model.h5                         # Saved DL model
│
├── requirements.txt
├── README.md
└── Diabetes_Prediction_Report.docx          # Full project report
```

---

## ⚙️ Installation

**1. Clone the repository**
```bash
git clone https://github.com/your-username/diabetes-prediction-system.git
cd diabetes-prediction-system
```

**2. Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Download the dataset**

Download from [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) and place `diabetes.csv` inside the `data/` directory.

**5. Run the notebooks**
```bash
jupyter notebook
```

### `requirements.txt`
```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
tensorflow>=2.10.0
matplotlib>=3.6.0
seaborn>=0.12.0
jupyter>=1.0.0
joblib>=1.2.0
```

---

## 🔬 Methodology

### Data Preprocessing

- **Missing value handling:** Zero values in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` are biologically implausible and were replaced with **column medians**
- **Feature scaling:** All features standardised using `StandardScaler` (zero mean, unit variance) — required for SVM, K-Means, and the neural network
- **Train-test split:** 80% training (614 samples) / 20% testing (154 samples) with `stratify=y` to preserve class proportions
- **Random seed:** `random_state=42` throughout for full reproducibility

---

### Machine Learning — Part A

Three classifiers were trained and evaluated on the binary diabetes classification task:

#### 🔹 Logistic Regression
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
```
- Serves as the linear baseline classifier
- Test accuracy: **~75–78%**

#### 🔹 Random Forest *(Best Model)*
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```
- Ensemble of 100 decision trees with bagging and random feature subsets
- Test accuracy: **~85–90%** — highest of all models
- Provides built-in **feature importance** scores

#### 🔹 Support Vector Machine (SVM)
```python
from sklearn.svm import SVC
model = SVC(kernel='rbf', random_state=42)
model.fit(X_train, y_train)
```
- RBF kernel captures non-linear decision boundaries
- Test accuracy: **~77–80%**

#### 🔹 Hyperparameter Tuning — Random Forest
```python
from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, None]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42),
                           param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
```

---

### Deep Learning — Part B

A **Multi-Layer Perceptron (MLP)** was built using TensorFlow/Keras:

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(8,)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    epochs=50,
                    batch_size=16,
                    validation_split=0.2)
```

| Layer | Units | Activation |
|-------|-------|------------|
| Dense (Input) | 64 | ReLU |
| Dropout | 0.3 | — |
| Dense (Hidden) | 32 | ReLU |
| Dropout | 0.2 | — |
| Dense (Output) | 1 | Sigmoid |

- Final test accuracy: **~78–80%**
- Training converges stably within 50 epochs with no overfitting (train ≈ val accuracy)

---

### Clustering — Part C

Unsupervised clustering was applied **without using the Outcome label** to discover natural patient risk groups:

```python
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Determine optimal K using silhouette analysis
scores = []
for k in range(2, 12):
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X_scaled)
    scores.append(silhouette_score(X_scaled, labels))

# Apply K-Means with optimal K=3
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
```

**Three clusters identified (K=3):**

| Cluster | Profile | Risk Level |
|---------|---------|------------|
| Cluster 1 | High Glucose & High BMI | 🔴 High Risk |
| Cluster 2 | Moderate Glucose & BMI | 🟡 Medium Risk |
| Cluster 3 | Low Glucose & Low BMI | 🟢 Low Risk |

---

## 📈 Results

### Model Performance Comparison

| Model | Type | Test Accuracy | Key Strength |
|-------|------|--------------|--------------|
| **Random Forest** ⭐ | ML | **~85–90%** | Highest accuracy; interpretable |
| Tuned Random Forest | ML | ~85–89% | Cross-validated; robust |
| Deep Learning MLP | DL | ~78–80% | Automatic feature learning |
| SVM (RBF Kernel) | ML | ~77–80% | Strong on structured data |
| Logistic Regression | ML | ~75–78% | Fast linear baseline |

### Feature Importance (Random Forest)

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | Glucose | ⬛⬛⬛⬛⬛⬛⬛⬛ Highest |
| 2 | BMI | ⬛⬛⬛⬛⬛⬛ High |
| 3 | Age | ⬛⬛⬛⬛⬛ Moderate |
| 4 | DiabetesPedigreeFunction | ⬛⬛⬛⬛ Moderate |
| 5 | Insulin | ⬛⬛⬛ Low-Moderate |
| 6 | Pregnancies | ⬛⬛ Low |
| 7 | BloodPressure | ⬛ Lowest |
| 8 | SkinThickness | ⬛ Lowest |

> ✅ **Best Model: Random Forest** — highest accuracy, feature importance for clinical interpretability, and robust generalisation.

---

## 🖼️ Visualisations

### Feature Correlation Heatmap
![Feature Correlation Heatmap](images/Feature_Correlation_Heatmap.jpeg)
> *Glucose (0.49) and BMI (0.31) show the strongest positive correlations with the diabetic Outcome.*

---

### Patient Clusters — Glucose vs BMI
![Patient Clusters](images/Patient_Clusters_Glucose_vs_BMI.jpeg)
> *K-Means (K=3) cleanly separates patients into high-risk, medium-risk, and low-risk groups based on Glucose and BMI.*

---

### Confusion Matrices

| Logistic Regression | Random Forest |
|---|---|
| ![LR](images/Logistic_Regression_Confusion_Matrix.jpeg) | ![RF](images/Random_Forest_Confusion_Matrix.jpeg) |

| Tuned Random Forest | SVM |
|---|---|
| ![Tuned RF](images/Tuned_Random_Forest_Confusion_Matrix.jpeg) | ![SVM](images/SVM_Confusion_Matrix.jpeg) |

---

### Deep Learning — Training & Validation Accuracy
![DL Accuracy](images/Deep_Learning_Accuracy_Curve.jpeg)
> *Training and validation accuracy converge near 80% by epoch 50, with no signs of overfitting.*

---

## 💡 Key Findings

- All models significantly outperform a random-chance baseline of 50%, confirming the 8 clinical features are highly predictive of diabetes
- **Glucose** is the single most important predictor — consistent with its central role in diabetes diagnosis
- **Random Forest** achieves the best performance (~85–90%) and is recommended for production deployment due to its accuracy, interpretability, and speed
- The **Deep Learning MLP** (~78–80%) is competitive but does not surpass ensemble methods at this data scale — consistent with the broader ML literature on tabular datasets
- **K-Means clustering (K=3)** successfully stratifies patients into three clinically meaningful risk groups without using the diagnosis label — enabling unsupervised risk triage
- The persistent misclassification of borderline cases across all models reflects genuine physiological overlap between pre-diabetic and diabetic patient profiles

---

## 🔮 Future Work

- [ ] **SHAP explainability** — patient-level feature contribution scores for clinical consultation
- [ ] **LSTM/RNN** — longitudinal glucose monitoring and time-series risk prediction
- [ ] **CNN on retinal imaging** — early diabetic retinopathy detection
- [ ] **REST API deployment** — Flask/FastAPI endpoint integrated with electronic health records
- [ ] **IoT integration** — real-time risk scoring from wearable glucose monitors
- [ ] **DBSCAN clustering** — identify irregular cluster shapes and anomalous patient profiles

---

## 📚 References

- Breiman, L. (2001) *Random Forests*, Machine Learning, 45(1), pp. 5–32.
- Cortes, C. and Vapnik, V. (1995) *Support-vector networks*, Machine Learning, 20(3), pp. 273–297.
- Chollet, F. (2021) *Deep Learning with Python*. 2nd edn. Manning Publications.
- Goodfellow, I., Bengio, Y. and Courville, A. (2016) *Deep Learning*. MIT Press.
- Pedregosa, F. et al. (2011) *Scikit-learn: Machine Learning in Python*, JMLR, 12, pp. 2825–2830.
- NIDDK (2016) *Pima Indians Diabetes Dataset*. [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).

---

## 👤 Author

**Student No:** IT/CAI/26/03/0031
**Programme:** Certificate in Artificial Intelligence
**Domain:** Healthcare Analytics
**Submission:** 25th April 2026

---

<p align="center">
  <i>Built with Python • Scikit-Learn • TensorFlow • Matplotlib • Seaborn</i>
</p>
