import pandas as pd

df = pd.read_csv("dataset.csv")
    
print(df.shape)      # Rows x Columns

print(df.info())
print("\nMissing values:\n", df.isnull().sum())

import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='price_range', data=df, palette='Set2')
plt.title("Price Range Distribution")
plt.xlabel("0 = Low, 1 = Medium, 2 = High, 3 = Very High")
plt.ylabel("Count")
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(15, 10))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

from sklearn.preprocessing import StandardScaler

X = df.drop('price_range', axis=1)
y = df['price_range']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

log_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
log_model.fit(X_train, y_train)

log_pred = log_model.predict(X_test)

print("\nLogistic Regression Results")
print("Accuracy:", accuracy_score(y_test, log_pred))
print("Classification Report:\n", classification_report(y_test, log_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, log_pred))


#Random forest
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("\nRandom Forest Results")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Classification Report:\n", classification_report(y_test, rf_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))

#xgBoost
from xgboost import XGBClassifier

xgb_model = XGBClassifier(objective='multi:softmax', num_class=4, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)

print("\nXGBoost Results")
print("Accuracy:", accuracy_score(y_test, xgb_pred))
print("Classification Report:\n", classification_report(y_test, xgb_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, xgb_pred))

import numpy as np
import matplotlib.pyplot as plt

coefficients = log_model.coef_[0]  # Use one class for visualizing
feature_names = X.columns

# Plot
plt.figure(figsize=(10, 6))
plt.barh(feature_names, coefficients)
plt.title("Feature Influence on Price Class 0 (Logistic Regression)")
plt.xlabel("Coefficient Value")
plt.tight_layout()
plt.show()



import joblib

joblib.dump(log_model, 'mobile_price_model.pkl')

loaded_model = joblib.load('mobile_price_model.pkl')
pred = loaded_model.predict(X_test)
