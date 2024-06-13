# Heart Attack Classification using K Neighbors Classifier

## Project Overview
This project aims to predict the likelihood of heart attacks using machine learning techniques. Specifically, it employs the K Neighbors Classifier algorithm on the "Heart Attack Analysis & Prediction Dataset." The goal is to analyze and predict heart attack risks based on various medical attributes.

### Authors
- Marouan Daghmoumi
- Abderrazzak El Bourkadi
- Chaimae Chouaa

### Supervisors
- Pr. Abdellah AZMANI
- Loubna Bouhsaien

## Dataset Description
The "Heart Attack Analysis & Prediction Dataset" includes the following variables:

- **Age**: Age of the patient
- **Sex**: Sex of the patient (1 = male; 0 = female)
- **exang**: Exercise induced angina (1 = yes; 0 = no)
- **ca**: Number of major vessels colored by fluoroscopy (0-3)
- **cp**: Chest pain type
  - Value 1: Typical angina
  - Value 2: Atypical angina
  - Value 3: Non-anginal pain
  - Value 4: Asymptomatic
- **trtbps**: Resting blood pressure (in mm Hg)
- **chol**: Cholesterol level in mg/dl
- **fbs**: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
- **restecg**: Resting electrocardiographic results
  - Value 0: Normal
  - Value 1: Having ST-T wave abnormality (T wave inversions and/or ST elevation or depression > 0.05 mV)
  - Value 2: Showing probable or definite left ventricular hypertrophy by Estes' criteria
- **thalach**: Maximum heart rate achieved
- **target**: Heart attack risk (0 = less likely; 1 = more likely)

## Objective
The objective is to leverage machine learning to classify heart attack risks. By analyzing and identifying patterns in the dataset, we aim to build an accurate predictive model for heart attack classification.

## Project Steps

### 1. Importing Libraries
```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
```
2. Data Importation
```python
data = pd.read_csv('heart.csv')
print(data)
```
3. Data Exploration and Visualization
```python
# Display first and last 10 rows of the dataset
data.head(10)
data.tail(10)
# Display dataset dimensions
data.shape
# Count examples with output 0 and 1
```python
count_output_0 = (data['output'] == 0).sum()
count_output_1 = (data['output'] == 1).sum()
# Display counts and distribution of target variable
```python
print(f"Le nombre d'exemples avec output égal à 0 est : {count_output_0}")
print(f"Le nombre d'exemples avec output égal à 1 est : {count_output_1}")
# Plot target variable distribution
plt.figure(figsize=(8, 5))
sns.countplot(x=data['output'])
plt.title("Distribution des valeurs de la variable cible", size=12)
plt.show()
```
4. Data Visualization
```python
# Boxplot for each feature
col = data.columns
fig, ax = plt.subplots(len(data.columns), 1, figsize=(8, 55))
for ind, axi in enumerate(ax.flat):
    axi.boxplot(data[col[ind]], vert=False)
    axi.set_title(col[ind], size=12)
plt.show()

# Correlation heatmap
cor = data.corr()
plt.figure(figsize=(12, 9))
sns.heatmap(cor, annot=True)
plt.show()
# Important features based on correlation with target variable
rel = cor['output'].sort_values(ascending=False)
print(rel)
```
5. Data Preparation
```python
# Handling missing values
values = data.isnull().sum()
print(values)

# Standardizing the features
X = data.drop('output', axis=1)
y = data['output']
col = X.columns
std = StandardScaler()
X = std.fit_transform(X)
X = pd.DataFrame(data=X, columns=col)

# Splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
6. Model Training and Evaluation
```python
# Initializing and training the KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Making predictions
y_pred = knn.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{class_report}")
```
Conclusion
This project demonstrates the use of K Neighbors Classifier to predict heart attack risks based on various medical attributes. By following the steps outlined, we successfully trained and evaluated a model capable of classifying individuals based on their heart attack risk.

License
This project is licensed under the MIT License.
