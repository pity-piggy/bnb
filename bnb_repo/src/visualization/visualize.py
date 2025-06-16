import pandas as pd
import numpy as np
import seaborn as sns
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def load_data_and_model():
    BASE_DIR = Path(__file__).resolve().parents[2]
    MODELS_DIR = BASE_DIR / "models"
    data_path = BASE_DIR / "data" / "processed" / "airbnb_montreal_cleaned.csv"

    models = {}
    for model_file in MODELS_DIR.glob("*_model.joblib"):
        name = model_file.stem.split("_")[0]
        models[name] = joblib.load(model_file)

    df = pd.read_csv(data_path)
    df = df[df['price'] < 1000]
    X = df.drop(columns=["log_price", "price", "log_price_per_person"])
    y = df["log_price_per_person"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return models, X_test, y_test, X.columns


def plot_pred_vs_actual(model, X_test, y_test, save_path=None):
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_actual = np.expm1(y_test)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=y_actual, y=y_pred, alpha=0.4, ax=ax, color="steelblue")

    min_val = min(y_actual.min(), y_pred.min())
    max_val = max(y_actual.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Perfect Prediction")

    ax.set_title("Predicted vs Actual Airbnb Prices", fontsize=14, pad=15)
    ax.set_xlabel("Actual Price ($)", fontsize=12, labelpad=10)
    ax.set_ylabel("Predicted Price ($)", fontsize=12, labelpad=10)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
        print(f"Saved prediction plot to {save_path}")
    else:
        plt.show()


def plot_residuals(model, X_test, y_test, save_path=None):
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_actual = np.expm1(y_test)
    residuals = y_actual - y_pred

    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=y_actual, y=residuals, alpha=0.4, color="tomato")
    plt.axhline(0, linestyle="--", color="gray")
    plt.title("Residuals vs Actual Price", fontsize=14)
    plt.xlabel("Actual Price ($)", fontsize=12)
    plt.ylabel("Residual (Actual - Predicted)", fontsize=12)
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved residual plot to {save_path}")
    else:
        plt.show()


def plot_feature_importance(model, feature_names, model_name, save_path=None):
    if not hasattr(model, "feature_importances_"):
        print(f"Model {model_name} has no feature_importances_ attribute.")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices][:15], y=np.array(feature_names)[indices][:15], palette="viridis")
    plt.title(f"Top 15 Feature Importances: {model_name.upper()}", fontsize=14)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved feature importance plot to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[2]
    FIGURE_DIR = BASE_DIR / "reports" / "figures"
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    models, X_test, y_test, feature_names = load_data_and_model()

    for name, model in models.items():
        print(f"\nEvaluating model: {name.upper()}")

        y_pred_log = model.predict(X_test)
        y_pred = np.expm1(y_pred_log)
        y_actual = np.expm1(y_test)

        rmse_log = np.sqrt(mean_squared_error(y_test, y_pred_log))
        rmse_dollar = np.sqrt(mean_squared_error(y_actual, y_pred))
        print(f"RMSE (log): {rmse_log:.3f} | RMSE ($): {rmse_dollar:.2f}")

        plot_pred_vs_actual(model, X_test, y_test, save_path=FIGURE_DIR / f"{name}_pred_vs_actual.png")
        plot_residuals(model, X_test, y_test, save_path=FIGURE_DIR / f"{name}_residuals.png")
        plot_feature_importance(model, feature_names, name, save_path=FIGURE_DIR / f"{name}_feature_importance.png")
