import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
data = pd.read_csv('dataset/heart.csv')

# Feature selection
X = data[['age', 'chol', 'cp', 'thalach']]
y = data['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
with open('model/heart_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved successfully!")
