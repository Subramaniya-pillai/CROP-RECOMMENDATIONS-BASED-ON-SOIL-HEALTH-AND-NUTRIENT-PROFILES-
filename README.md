# CROP-RECOMMENDATIONS-BASED-ON-SOIL-HEALTH-AND-NUTRIENT-PROFILES-

The Crop Recommendation System uses machine learning models (Random Forest, KNN, HistGB) to suggest optimal crops based on soil health parameters like N, P, K, pH, temperature, humidity, and rainfall. It provides data-driven insights to help farmers maximize yield and maintain soil health. The system features a user-friendly interface for easy access and interpretation. Future enhancements include real-time weather integration and mobile application development for precision agriculture.


## Features:

1.  Data-Driven Crop Recommendation – Suggests optimal crops based on soil health parameters.
2.  Machine Learning Models – Utilizes Random Forest, KNN, and HistGB for accurate predictions.
3.  Soil and Environmental Analysis – Evaluates N, P, K, pH, temperature, humidity, and rainfall for recommendations.
4.  user-Friendly Interface – Simplifies access to crop suggestions for farmers of all technical levels.




## Requirements

1. Programming Language:Python 3.x: Used for the development of the project.
2. Essential Packages :pandas,numpy,scikit-learn,matplotlib,seaborn.

## Usage:

1. Clone the Repository
2. Install Dependencies
3. Run the Jupyter Notebook
4. Run the Web Interface
5. Input Data & Get Predictions

## System Architecture
![WhatsApp Image 2025-03-01 at 23 05 00_1bbc72a0](https://github.com/user-attachments/assets/1175977d-30e0-4ca6-aa82-714a9ff25ddc)


## Program:
```python
# Import necessary libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
import joblib

# Data Preprocessing
label_encoder = LabelEncoder()
data['Crop'] = label_encoder.fit_transform(data['Crop'])

X = data.drop("Crop", axis=1)
y = data["Crop"]

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"AUC: {roc_auc_score(y_test, y_proba, multi_class='ovr'):.2f}")
print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")

# K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
y_proba = knn.predict_proba(X_test)
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"AUC: {roc_auc_score(y_test, y_proba, multi_class='ovr'):.2f}")
print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")

# HistGradientBoosting Classifier
hist_gb = HistGradientBoostingClassifier(random_state=42)
hist_gb.fit(X_train, y_train)
y_pred = hist_gb.predict(X_test)
y_proba = hist_gb.predict_proba(X_test)
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"AUC: {roc_auc_score(y_test, y_proba, multi_class='ovr'):.2f}")
print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")

# Ensemble Model
ensemble_model = VotingClassifier(estimators=[
    ('rf', rf),
    ('knn', knn),
    ('hist_gb', hist_gb)
], voting='soft')
ensemble_model.fit(X_train, y_train)

y_pred = ensemble_model.predict(X_test)
y_proba = ensemble_model.predict_proba(X_test)
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"AUC: {roc_auc_score(y_test, y_proba, multi_class='ovr'):.2f}")
print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")

# Save Model
joblib.dump(ensemble_model, 'model_crops.pkl')


```
## Output:
CROP-RECOMMENDATIONS-BASED-ON-SOIL-HEALTH-AND-NUTRIENT-PROFILES

![WhatsApp Image 2025-03-01 at 00 23 02_806631c8](https://github.com/user-attachments/assets/cc4200e9-4912-4e70-bd31-6f4951abe4c3)

## Result:
1.  The system provides personalized crop suggestions by analyzing soil health and environmental factors, ensuring higher yield and efficient resource utilization.

2.  By recommending crops based on soil nutrient profiles, the system helps in preserving soil fertility and promoting sustainable agriculture.

3.  The user-friendly interface and scientific guidance enable farmers, especially those in rural areas, to make informed, data-driven decisions for better productivity and economic stability.

## Articles published / References:
1.  Ayesha Barvin, P.; Sampradeepraj, T. Crop Recommendation Systems Based on Soil and Environmental Factors Using Graph Convolution Neural Network: A Systematic Literature Review. Eng. Proc. 2023, 58, 97. https://doi.org/10.3390/ecsa-10-16010
2.  M. R, A. Kumar K, C. K. N, D. A and A. T, "Soil Nutrient Prediction and Crop Recommendation System," 2023 International Conference on Circuit Power and Computing Technologies (ICCPCT), Kollam, India, 2023, pp. 976-981, doi: 10.1109/ICCPCT58313.2023.10245685.
3.  Sivanandhini, P.; Prakash, J. Crop Yield Prediction Analysis using Feed Forward and Recurrent Neural Network. Int. J. Innov. Sci. Res. Technol. 2020, 5, 1092–1096.
4.  Pruthviraj; Akshatha, G.C.; Shastry, K.A.; Nagaraj; Nikhil. Crop and fertilizer recommendation system based on soil classification. In Recent Advances in Artificial Intelligence and Data Engineering, Proceedings of AIDE 2020, Karkala, India, 22–23 December 2020; Springer: Berlin/Heidelberg, Germany, 2022; pp. 29–40
