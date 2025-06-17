import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import mlflow
from mlflow.sklearn import load_model

# --- CONFIGURATION ---
REGISTERED_MODEL_NAME = "airbnb_best_model_in_range"
MODEL_VERSION = 1
TARGET_COLUMN = "log_price"
DATA_FILE = "airbnb_montreal_cleaned.csv"

# Set MLflow tracking URI
mlflow.set_tracking_uri("file:///Users/chloe/PycharmProjects/bnb/bnb_repo/mlruns")

def get_base_dir():
    try:
        return Path(__file__).resolve().parents[2]
    except NameError:
        return Path.cwd()

def load_model_and_data():
    BASE_DIR = get_base_dir()
    DATA_PATH = BASE_DIR / "data" / "processed" / DATA_FILE

    model_uri = f"models:/{REGISTERED_MODEL_NAME}/{MODEL_VERSION}"
    model = load_model(model_uri)

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["log_price", "log_price_per_person", "price","price_per_person"], errors="ignore")
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return model, X_test, y_test, X.columns

def plot_pred_vs_actual(model, X_test, y_test, save_path=None):
    y_pred = model.predict(X_test)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.4, color="steelblue", ax=ax)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_title(" Predicted vs Actual Prices")
    ax.set_xlabel("Actual Price ($)")
    ax.set_ylabel("Predicted Price ($)")
    ax.grid(True)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
        print(f" Saved to {save_path}")
    else:
        plt.show()

def plot_residuals(model, X_test, y_test, save_path=None):
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=y_test, y=residuals, alpha=0.4, color="tomato")
    plt.axhline(0, linestyle="--", color="gray")
    plt.title("Residuals vs Actual Price")
    plt.xlabel("Actual Price ($)")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f" Saved residual plot to {save_path}")
    else:
        plt.show()

def plot_feature_importance(model, feature_names, save_path=None):
    if not hasattr(model, "feature_importances_"):
        print("Model does not support feature_importances_")
        return
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices][:15],
                y=np.array(feature_names)[indices][:15],
                palette="viridis")
    plt.title("Top 15 Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f" Saved feature importance plot to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    BASE_DIR = get_base_dir()
    FIGURE_DIR = BASE_DIR / "reports" / "figures"
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    model, X_test, y_test, feature_names = load_model_and_data()
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"\n RMSE of '{REGISTERED_MODEL_NAME}' (v{MODEL_VERSION}): ${rmse:.2f}")

    plot_pred_vs_actual(
        model, X_test, y_test,
        save_path=FIGURE_DIR / f"{REGISTERED_MODEL_NAME}_pred_vs_actual.png"
    )

    plot_residuals(
        model, X_test, y_test,
        save_path=FIGURE_DIR / f"{REGISTERED_MODEL_NAME}_residuals.png"
    )

    plot_feature_importance(
        model, feature_names,
        save_path=FIGURE_DIR / f"{REGISTERED_MODEL_NAME}_feature_importance.png"
    )
