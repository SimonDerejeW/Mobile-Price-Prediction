import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model_implementation import model

# Read the data
data = pd.read_csv('data/Cellphone.csv')

# Remove 'Product_id' column since it's not needed
data.drop('Product_id', inplace=True, axis=1)

# Separate independent and dependent variables
X = data.drop("Price", axis=1)
y = data["Price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Split training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Standardize features
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_val = scaler_X.transform(X_val)
X_test = scaler_X.transform(X_test)

# Initialize and find best learning rate using validation set
print("\nFinding best learning rate...")
model = model.GradientDescentMultipleRegression(n_iterations=1000)
best_lr, results = model.find_best_learning_rate(X_train, y_train, X_val, y_val)
print(f"Best learning rate found: {best_lr}")

# Train final model with best learning rate on full training data
X_train_full = np.vstack([X_train, X_val])
y_train_full = np.concatenate([y_train, y_val])

print("\nTraining final model with best learning rate...")
model.learning_rate = best_lr
model.fit(X_train_full, y_train_full)

# Save both the trained model and the scaler
joblib.dump(model, 'model.pkl')
joblib.dump(scaler_X, 'scaler.pkl')
