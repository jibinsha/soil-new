import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
import joblib

# Load dataset
data = pd.read_csv("cleaned_ai_training_data.csv")

# Features (X)
X = data[
    ['LATITUDE', 'LONGITUDE', 'Elevation', 'Slope', 'Rainfall',
     'MinTemp', 'MaxTemp', 'Humidity', 'NDVI']
]

# Targets (y)
y = data[
    ['N', 'P', 'K', 'Ca', 'Mg', 'S', 'Zn',
     'Fe', 'Mn', 'Cu', 'B', 'pH', 'EC', 'Org_C']
]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
base_model = XGBRegressor(n_estimators=200, learning_rate=0.05)
model = MultiOutputRegressor(base_model)

# Train
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "soil_model.pkl")

print("Model trained successfully 🚀")