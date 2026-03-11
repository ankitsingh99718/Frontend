import pandas as pd
import os
import joblib
from sklearn.linear_model import LinearRegression

# CSV file path
file_path = os.path.join(os.getcwd(), "nifty_50.csv")

# CSV load
data = pd.read_csv(file_path)

# Sirf Open column use karna
X = data[['Open']]

# Next day open price predict karna
y = data['Open'].shift(-1)

# Last row remove (kyunki next value nahi hoti)
X = X[:-1]
y = y[:-1]

# Model create
model = LinearRegression()

# Model train
model.fit(X, y)

# Model save
model_path = os.path.join(os.getcwd(), "model.pkl")
joblib.dump(model, model_path)

print("Model trained successfully!")
print("Model saved at:", model_path)