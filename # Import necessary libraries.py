# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Create synthetic dataset
np.random.seed(0)
data_size = 1000
data = pd.DataFrame({
    'Income': np.random.randint(20000, 120000, data_size),
    'Credit_Score': np.random.randint(300, 850, data_size),
    'Loan_Amount': np.random.randint(5000, 50000, data_size),
    'Loan_Term': np.random.choice([15, 30], data_size),
    'Eligibility': np.random.choice([0, 1], data_size)  # 0: Not Eligible, 1: Eligible
})

# Display the first few rows of the dataset
print(data.head())

# Data Preprocessing
X = data.drop('Eligibility', axis=1)
y = data['Eligibility']

# Encode categorical variables
X = pd.get_dummies(X, columns=['Loan_Term'], drop_first=True)

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print accuracy and classification report
print(f'Accuracy: {accuracy:.2f}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{class_report}')

# Visualization of the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Eligible', 'Eligible'], yticklabels=['Not Eligible', 'Eligible'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
