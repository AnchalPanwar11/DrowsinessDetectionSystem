import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# load dataset
data = pd.read_csv("driver_data.csv")

# features and labels
X = data[["ml_score", "tilt", "distance"]]
y = data["label"]

# create model
model = RandomForestClassifier(n_estimators=100)

# train
model.fit(X, y)

# save model
joblib.dump(model, "risk_model.pkl")

print("✅ Model trained and saved!")