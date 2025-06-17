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
import mlflow.sklearn
from mlflow.models.signature import infer_signature



from sklearn.model_selection import cross_val_score

# ... (keep all your imports the same)

def train_and_log(data_path, target_column, experiment_name):
    mlflow.set_tracking_uri("file:///Users/chloe/PycharmProjects/bnb/bnb_repo/mlruns")
    mlflow.set_experiment(experiment_name)


    # Load Data
    df = pd.read_csv(data_path)
    X = df.drop(columns=["log_price", "price", "log_price_per_person","price_per_person"], errors="ignore")
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

    BASE_DIR = Path(__file__).resolve().parents[2]
    FIGURE_DIR = BASE_DIR / "reports" / "figures"
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "models").mkdir(exist_ok=True)

    results = []
    is_log_target = target_column.startswith("log_")

    for name, cfg in model_configs.items():
        print(f"\nTraining model: {name.upper()}")

        # Grid Search
        grid = GridSearchCV(cfg["model"], cfg["params"], cv=4,
                            scoring="neg_root_mean_squared_error", verbose=1, n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        # Predictions
        y_pred = best_model.predict(X_test)
        if is_log_target:
            y_pred_price = np.expm1(y_pred)
            y_test_price = np.expm1(y_test)
        else:
            y_pred_price = y_pred
            y_test_price = y_test

        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        real_rmse = np.sqrt(mean_squared_error(y_test_price, y_pred_price))

        print(f"Best Params for {name}: {grid.best_params_}")
        print(f"R2: {r2:.3f} | RMSE: {rmse:.3f} | MAE: {mae:.3f} | Real RMSE ($): {real_rmse:.2f}")

        # Log to MLflow
        with mlflow.start_run(run_name=name.upper()):
            mlflow.log_params(grid.best_params_)
            mlflow.log_metrics({
                "rmse_raw": rmse,
                "r2_score": r2,
                "mae_log": mae,
                "rmse_dollar": real_rmse
            })
            mlflow.set_tag("dataset_used", Path(data_path).name)  # Tag: "airbnb_montreal_cleaned.csv"
            mlflow.log_artifact(data_path, artifact_path="input_data")  # Optional: Log actual file

            input_example = X_test.iloc[:5]
            signature = infer_signature(X_test, y_pred)

            mlflow.sklearn.log_model(
                sk_model=best_model,
                name="model",
                input_example=input_example,
                signature=signature
            )

        print(f"Logged {name} model to MLflow.")




    # Optional: Save results
    # pd.DataFrame(results).to_csv(BASE_DIR / "models" / "model_comparison.csv", index=False)



if __name__ == "__main__":
    train_and_log("/Users/chloe/PycharmProjects/bnb/bnb_repo/data/processed/airbnb_montreal_cleaned.csv",
                  "price",
                  "airbnb_pricing_models")
    train_and_log("/Users/chloe/PycharmProjects/bnb/bnb_repo/data/processed/airbnb_montreal_cleaned.csv",
                  "log_price_per_person",
                  "airbnb_pricing_models_1")
    train_and_log("/Users/chloe/PycharmProjects/bnb/bnb_repo/data/processed/airbnb_montreal_cleaned.csv",
                  "log_price",
                  "airbnb_pricing_models_2")
    train_and_log("/Users/chloe/PycharmProjects/bnb/bnb_repo/data/processed/airbnb_montreal_cleaned_1.csv",
                  "log_price",
                  "airbnb_pricing_models_3")
