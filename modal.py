# ==============================
# 1. IMPORT LIBRARIES
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ML Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Clustering
from sklearn.cluster import KMeans, AgglomerativeClustering

# Deep Learning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# ==============================
# 2. LOAD DATA
# ==============================
df = pd.read_csv("diabetes.csv")

# ==============================
# 3. PREPROCESSING
# ==============================

# Replace zeros with NaN
cols = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
df[cols] = df[cols].replace(0, np.nan)

# Fill missing values
df.fillna(df.median(), inplace=True)

# Features & target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ==============================
# 4. VISUALIZATION
# ==============================

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# ==============================
# 5. MACHINE LEARNING
# ==============================

def evaluate(name, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(name)
    plt.show()

# Models
evaluate("Logistic Regression", LogisticRegression())
evaluate("Random Forest", RandomForestClassifier())
evaluate("SVM", SVC())

# ==============================
# 6. HYPERPARAMETER TUNING
# ==============================

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6, 10]
}

grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=3)
grid.fit(X_train, y_train)

best_rf = grid.best_estimator_
evaluate("Tuned Random Forest", best_rf)

# ==============================
# 7. DEEP LEARNING
# ==============================

model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(X_train, y_train, epochs=50, validation_split=0.2)

plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.legend()
plt.title("DL Accuracy")
plt.show()

loss, acc = model.evaluate(X_test, y_test)
print("DL Accuracy:", acc)

# ==============================
# 8. CLUSTERING
# ==============================

kmeans = KMeans(n_clusters=3)
df['Cluster'] = kmeans.fit_predict(X_scaled)

sns.scatterplot(x=df['Glucose'], y=df['BMI'], hue=df['Cluster'])
plt.title("Patient Clusters")
plt.show()

print(df.groupby('Cluster').mean())