import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import zipfile
import urllib.request

# Download the ZIP file
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
zip_path, _ = urllib.request.urlretrieve(url)

# Extract the CSV file from the ZIP archive
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall()

# Load the Dataset
bank_data = pd.read_csv("bank-additional/bank-additional-full.csv", sep=';')

# Check data types
print("Data Types:")
print(bank_data.dtypes)

# Check for missing values
print("\nMissing Values:")
print(bank_data.isnull().sum())

# Data Preprocessing
X = bank_data.drop(columns=['y'])
y = (bank_data['y'] == 'yes').astype(int)  # Convert target variable to binary

# Convert categorical variables to dummy/indicator variables
X = pd.get_dummies(X, drop_first=True)

# Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the Decision Tree Classifier
clf = DecisionTreeClassifier(max_depth=5, random_state=42)

# Train the Model
clf.fit(X_train, y_train)

# Evaluate the Model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

print("\nAccuracy:", accuracy)
print("\nClassification Report:")
print(report)
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nROC AUC Score:", roc_auc)
