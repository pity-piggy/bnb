import os
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

# === Load and Prepare Data ===
print(" Loading processed data...")
df = pd.read_csv("../../data/processed/airbnb_montreal_cleaned.csv")

# Feature/target split
X = df.drop(columns=["log_price","price","log_price_per_person"])
y = df["log_price"]

print(f" Dataset shape: {X.shape}")
print(f" Target variable mean: {y.mean():.2f}")

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(" Data split: ", X_train.shape, X_test.shape)

# === Hyperparameter Tuning ===
print(" Starting hyperparameter tuning...")
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0]
}

grid_search = GridSearchCV(
    estimator=XGBRegressor(objective="reg:squarederror", random_state=42),
    param_grid=param_grid,
    scoring="neg_root_mean_squared_error",
    cv=3,
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f" Best Parameters: {grid_search.best_params_}")

# === Evaluation ===
train_rmse = np.sqrt(mean_squared_error(y_train, best_model.predict(X_train)))
test_rmse = np.sqrt(mean_squared_error(y_test, best_model.predict(X_test)))

print(f"\n Training RMSE: {train_rmse:.2f}")
print(f"Test RMSE: {test_rmse:.2f}")

# === Save Model ===
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "/Users/chloe/PycharmProjects/bnb/bnb_repo/models/xgb_model.joblib")
print("Model saved to /Users/chloe/PycharmProjects/bnb/bnb_repo/models/xgb_model.joblib")

import matplotlib.pyplot as plt
import xgboost as xgb

xgb.plot_importance(best_model, max_num_features=20)
plt.show()
