import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import kagglehub

# Download latest version
path = kagglehub.dataset_download("sarahtaha/1025-pokemon")

print("Path to dataset files:", path)

pd.read_csv("gs://datasets-maxalfonso/all_pokemon_data.csv")

df = pd.read_csv("gs://datasets-maxalfonso/all_pokemon_data.csv")
print(df.shape)
df.head()

df.describe(include='all')

df.info()

df['Legendary Status'] = df['Legendary Status'].astype(int)

df.head()

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Define features (X) and target variable (y)
features = ['Name', 'Primary Typing', 'Generation', 'Form', 'Evolution Stage', 'Weight (hg)', 'Height (in)', 'Weight (lbs)', 'Base Stat Total', 'Health', 'Attack', 'Defense', 'Special Attack', 'Special Defense', 'Speed']
target = 'Legendary Status'

X = df[features]
y = df[target]

# One-hot encode categorical features
X = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a decision tree classifier
clf = DecisionTreeClassifier()

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


from sklearn.metrics import confusion_matrix, classification_report

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Generate classification report
cr = classification_report(y_test, y_pred)
print("\nClassification Report:\n", cr)

!pip install google-cloud-aiplatform
from google.cloud import aiplatform

PROJECT_ID = "maxalfonso-dev"
LOCATION = "southamerica-west1"
ENDPOINT_NAME = "pokemoncl"  # @param {type:"string"}
DISPLAY_NAME = "pokemoncl"  # @param {type:"string"}
INFERENCE_TIMEOUT_SECS = "100"



aiplatform.init(
    project=PROJECT_ID,
    location=LOCATION,
    
)

try:
    dedicated_endpoint = aiplatform.Endpoint.create(
      display_name=DISPLAY_NAME,
      dedicated_endpoint_enabled=True,
      sync=True,
      inference_timeout=int(INFERENCE_TIMEOUT_SECS),
)
except Exception as e:
    print(f"Error creating endpoint: {e}")
    print("Please check your project ID, location, and network configuration.")
    print("You may also want to try again later, as this could be a temporary issue.")
    raise 

