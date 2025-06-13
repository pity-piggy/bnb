import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor


import mlflow
import mlflow.sklearn  # or mlflow.xgboost if logging native model

mlflow.set_tracking_uri("file:///Users/chloe/PycharmProjects/bnb/bnb_repo/mlruns")
mlflow.set_experiment("airbnb_pricing_models")

# === STEP 1: Load Data ===
df = pd.read_csv("/Users/chloe/PycharmProjects/bnb/bnb_repo/data/processed/airbnb_montreal_cleaned.csv")
X = df.drop(columns=["log_price", "price", "log_price_per_person"])
y = df["log_price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === STEP 2: Define models and parameter grids ===
model_configs = {
    "xgb": {
        "model": XGBRegressor(objective="reg:squarederror", random_state=42),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8, 1.0]
        }
    },
    "rf": {
        "model": RandomForestRegressor(random_state=42),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [10, 20]
        }
    },
    "et": {
        "model": ExtraTreesRegressor(random_state=42),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [10, 20]
        }
    }
}

# === STEP 3: Loop through models ===
results = []
BASE_DIR = Path(__file__).resolve().parents[2]
FIGURE_DIR = BASE_DIR / "reports" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
Path("models").mkdir(exist_ok=True)

for name, cfg in model_configs.items():
    print(f"\n🚀 Training model: {name.upper()}")



    # Grid search
    grid = GridSearchCV(cfg["model"], cfg["params"], cv=3,
                        scoring="neg_root_mean_squared_error", verbose=1, n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_


    # Evaluation
    y_pred_log = best_model.predict(X_test)
    y_pred_price = np.expm1(y_pred_log)
    y_test_price = np.expm1(y_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred_log))
    r2 = r2_score(y_test, y_pred_log)
    mae = mean_absolute_error(y_test, y_pred_log)
    real_rmse = np.sqrt(mean_squared_error(y_test_price, y_pred_price))

    print(f"✅ Best Params for {name}: {grid.best_params_}")
    print(f"🧠 R2: {r2:.3f} | RMSE (log): {rmse:.3f} | MAE (log): {mae:.3f} | 💰 RMSE ($): {real_rmse:.2f}")

    # # Save model
    # joblib.dump(best_model, BASE_DIR / "models"/f"{name}_model.joblib")



    with mlflow.start_run(run_name=name.upper()):
        mlflow.log_params(grid.best_params_)

        mlflow.log_metric("rmse_log", rmse)
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("mae_log", mae)
        mlflow.log_metric("rmse_dollar", real_rmse)

        # Log model file

        import mlflow.models

        from mlflow.models.signature import infer_signature

        # Pick one example row from X_test for signature
        input_example = X_test.iloc[:5]
        signature = infer_signature(X_test, y_pred_log)

        mlflow.sklearn.log_model(
            sk_model=best_model,
            name="model",
            input_example=input_example,
            signature=signature
        )

        # # Optional: Log feature importance plot
        # if name in ["xgb", "rf", "et"]:
        #     plot_path = FIGURE_DIR / f"{name}_feature_importance.png"
        #     mlflow.log_artifact(str(plot_path))

    print(f"✅ Logged {name} model to MLflow.")

    # Store result
    results.append({
        "model": name,
        "r2_score": r2,
        "rmse_log": rmse,
        "mae_log": mae,
        "rmse_dollar": real_rmse,
        "best_params": grid.best_params_
    })

    # Plot feature importance (tree models only)
    importance = best_model.feature_importances_
    features = X.columns
    sorted_idx = np.argsort(importance)[::-1]
    plt.figure(figsize=(8, 5))
    plt.barh(features[sorted_idx], importance[sorted_idx])
    plt.gca().invert_yaxis()
    plt.title(f"{name.upper()} Feature Importance")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"{name}_feature_importance.png")
    plt.close()

# # === STEP 4: Save summary results ===
# results_df = pd.DataFrame(results)
# results_df.to_csv("models/model_comparison.csv", index=False)
# print("\n✅ All models trained. Comparison saved to models/model_comparison.csv")
