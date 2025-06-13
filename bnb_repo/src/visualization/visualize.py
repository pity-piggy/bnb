import pandas as pd
import numpy as np
import seaborn as sns
import joblib
from sklearn.metrics import mean_squared_error
from pathlib import Path
import matplotlib.pyplot as plt
import xgboost as xgb


def load_data_and_model():
    # Define paths
    BASE_DIR = Path(__file__).resolve().parents[2]
    model_path = BASE_DIR / "models" / "xgb_model.joblib"
    data_path = BASE_DIR / "data" / "processed" / "airbnb_montreal_cleaned.csv"

    # Load model
    model = joblib.load(model_path)

    # Load and split data
    df = pd.read_csv(data_path)
    df = df[df['price'] < 1000]
    X = df.drop(columns=["log_price","price","log_price_per_person"])
    y = df["log_price_per_person"]

    # Reuse same split as before
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return model, X_test, y_test


def plot_pred_vs_actual(model, X_test, y_test, save_path=None):
    # Predict and inverse-transform from log scale
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_actual = np.expm1(y_test)

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=y_actual, y=y_pred, alpha=0.4, ax=ax, color="steelblue")

    # Perfect prediction line
    min_val = min(y_actual.min(), y_pred.min())
    max_val = max(y_actual.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Perfect Prediction")

    # Labels and title
    ax.set_title("Predicted vs Actual Airbnb Prices", fontsize=14, pad=15)
    ax.set_xlabel("Actual Price ($)", fontsize=12, labelpad=10)
    ax.set_ylabel("Predicted Price ($)", fontsize=12, labelpad=10)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True)

    # Fix layout
    fig.tight_layout()

    # Save or show
    if save_path:
        fig.savefig(save_path, dpi=300)
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

#
# def plot_feature_importance(model, save_path=None, max_features=20):
#     # Get feature importances as a DataFrame
#     booster = model.get_booster()
#     importance = booster.get_score(importance_type='gain')
#     importance_df = pd.DataFrame({
#         'Feature': list(importance.keys()),
#         'Importance': list(importance.values())
#     }).sort_values(by='Importance', ascending=False).head(max_features)
#
#     # Plot
#     plt.figure(figsize=(10, 8))
#     bars = plt.barh(importance_df['Feature'], importance_df['Importance'], color='steelblue')
#     plt.xlabel("Gain-based Importance", fontsize=12)
#     plt.title(f"Top {max_features} Feature Importances", fontsize=14)
#     plt.gca().invert_yaxis()  # Most important on top
#
#     # Add value labels inside bars
#     for bar in bars:
#         width = bar.get_width()
#         plt.text(width - (width * 0.05),  # position left a bit
#                  bar.get_y() + bar.get_height() / 2,
#                  f"{width:.2f}",
#                  va='center', ha='right', fontsize=9, color='white')
#
#     plt.tight_layout()
#
#     if save_path:
#         plt.savefig(save_path, dpi=300)
#         print(f" Feature importance plot saved to {save_path}")
#     else:
#         plt.show()

def plot_residuals(model, X_test, y_test, save_path=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # Get predicted and actual prices
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_actual = np.expm1(y_test)

    residuals = y_actual - y_pred

    # Plot
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
        print(f" Residual plot saved to {save_path}")
    else:
        plt.show()



if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[2]
    FIGURE_DIR = BASE_DIR / "reports" / "figures"
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    model, X_test, y_test = load_data_and_model()
    # Predicted vs Actual Plot
    plot_pred_vs_actual(model, X_test, y_test, save_path=FIGURE_DIR /"pred_vs_actual.png")
    # Feature Importance Plot
    # plot_feature_importance(model, save_path=FIGURE_DIR /"feature_importance.png")

    plot_residuals(model, X_test, y_test, save_path=FIGURE_DIR / "residuals_plot.png")

