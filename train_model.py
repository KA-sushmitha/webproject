import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("chronic_kidney_disease_full.csv")  # Ensure this file is in the same folder

# Replace missing values
df.replace('?', np.nan, inplace=True)
df.fillna(method='ffill', inplace=True)

# Convert numerical columns
numeric_cols = ['age','bp','sg','al','su','bgr','bu','sc','sod','pot','hemo','pcv','wbcc','rbcc']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df.fillna(method='ffill', inplace=True)

# Label encode all categorical columns
le = LabelEncoder()
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = le.fit_transform(df[column])

# Split features and label
X = df.drop(columns='classification')
y = df['classification']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Save model
with open("ckd_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved as ckd_model.pkl ✅")
