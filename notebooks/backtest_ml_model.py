"""
backtest_ml_model_v2.py
PROPER Machine Learning Backtesting Model
- Trains Random Forest classifier on indicator features
- Tests on out-of-sample data
- Calculates feature importance (indicator weights)
- Reports actual ML model accuracy

Run: python notebooks/backtest_ml_model_v2.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nse_analyzer.utils.config import ConfigManager
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
import warnings
import json

warnings.filterwarnings("ignore")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ==================== Setup ====================

config_manager = ConfigManager()
data_dir = config_manager.get_data_dir()
indicators_dir = data_dir / "indicators"
output_dir = Path(__file__).parent
output_dir.mkdir(parents=True, exist_ok=True)

stocks = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "KOTAKBANK"]

print("=" * 80)
print("ML BACKTEST v2 - RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 80)

# ==================== Load Data ====================


def load_indicator_data(stock_name):
    """Load indicator CSV for a stock."""
    csv_path = indicators_dir / f"{stock_name}_indicators.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df["Date"] = pd.to_datetime(df["Date"])
        return df.sort_values("Date").reset_index(drop=True)
    return None


data = {}
for stock in stocks:
    df = load_indicator_data(stock)
    if df is not None:
        data[stock] = df
        print(f"✅ {stock}: {len(df)} rows loaded")

# ==================== Prepare Features and Target ====================


def prepare_ml_data(df, target_periods_ahead=1):
    """
    Prepare data for ML model

    Features: Indicator values (normalized)
    Target: 1 if price goes up in next 'target_periods_ahead', 0 if goes down
    """
    df = df.copy()

    # Create target: predict if price goes up in next period
    df["Future_Close"] = df["Close"].shift(-target_periods_ahead)
    df["Target"] = (df["Future_Close"] > df["Close"]).astype(int)

    # Select indicator features
    features = [
        "RSI_14",
        "MACD",
        "MACD_Signal",
        "MACD_Histogram",
        "Supertrend_Direction",
        "BB_Upper_20",
        "BB_Middle_20",
        "BB_Lower_20",
        "BB_Percent_B_20",
    ]

    # Check which features exist
    available_features = [f for f in features if f in df.columns]

    # Remove NaN rows
    df_clean = df[available_features + ["Target"]].dropna()

    if len(df_clean) == 0:
        return None, None, None

    X = df_clean[available_features]
    y = df_clean["Target"]

    return X, y, available_features


# ==================== Train ML Models ====================

print("\n" + "=" * 80)
print("TRAINING RANDOM FOREST MODELS")
print("=" * 80)

all_feature_importance = {}
all_metrics = {}

for stock in stocks:
    print(f"\n{stock}:")

    if stock not in data:
        print("  ❌ No data available")
        continue

    X, y, features = prepare_ml_data(data[stock])

    if X is None or len(X) < 50:
        print(f"  ⚠️  Insufficient data ({len(X) if X is not None else 0} samples)")
        continue

    # Time series split (80-20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)

    # Predictions
    y_pred_train = rf.predict(X_train_scaled)
    y_pred_test = rf.predict(X_test_scaled)
    y_pred_proba = rf.predict_proba(X_test_scaled)[:, 1]

    # Metrics
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test, zero_division=0)
    recall = recall_score(y_test, y_pred_test, zero_division=0)
    f1 = f1_score(y_test, y_pred_test, zero_division=0)

    auc = roc_auc_score(y_test, y_pred_proba)

    print(f"  Train Accuracy: {train_acc*100:.2f}%")
    print(f"  Test Accuracy:  {test_acc*100:.2f}%")
    print(f"  Precision:      {precision*100:.2f}%")
    print(f"  Recall:         {recall*100:.2f}%")
    print(f"  F1 Score:       {f1*100:.2f}%")
    print(f"  AUC-ROC:        {auc*100:.2f}%")

    # Feature importance
    importance = pd.Series(rf.feature_importances_, index=features).sort_values(
        ascending=False
    )
    print("\n  Feature Importance:")
    for feat, imp in importance.items():
        print(f"    {feat:20} | {imp*100:6.2f}%")

    all_feature_importance[stock] = importance
    all_metrics[stock] = {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "n_test_samples": len(y_test),
    }

# ==================== Calculate Cumulative Weights ====================

print("\n" + "=" * 80)
print("CUMULATIVE INDICATOR WEIGHTS (From ML Feature Importance)")
print("=" * 80)

# Map feature names to indicator groups
feature_to_indicator = {
    "RSI_14": "RSI",
    "MACD": "MACD",
    "MACD_Signal": "MACD",
    "MACD_Histogram": "MACD",
    "Supertrend_Direction": "Supertrend",
    "BB_Upper_20": "BB",
    "BB_Middle_20": "BB",
    "BB_Lower_20": "BB",
    "BB_Percent_B_20": "BB",
}

# Aggregate importance by indicator
indicator_importance = {}

for stock, importance in all_feature_importance.items():
    for feature, imp_value in importance.items():
        indicator = feature_to_indicator.get(feature, "Other")
        if indicator not in indicator_importance:
            indicator_importance[indicator] = []
        indicator_importance[indicator].append(imp_value)

# Calculate average importance per indicator
avg_importance = {}
for indicator, values in indicator_importance.items():
    avg_importance[indicator] = np.mean(values)

# Normalize to sum to 1.0
total_imp = sum(avg_importance.values())
normalized_weights = {k: v / total_imp for k, v in avg_importance.items()}

print("\nRaw Feature Importance by Indicator (Average across stocks):")
for indicator, weight in sorted(
    avg_importance.items(), key=lambda x: x[1], reverse=True
):
    print(f"  {indicator:15} | {weight*100:6.2f}%")

print("\nNormalized Weights (Sum = 100%):")
for indicator, weight in sorted(
    normalized_weights.items(), key=lambda x: x[1], reverse=True
):
    print(f"  {indicator:15} | {weight*100:6.2f}%")

# ==================== Visualizations ====================

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Test Accuracy by Stock
ax1 = axes[0, 0]
stocks_tested = [s for s in stocks if s in all_metrics]
test_accs = [all_metrics[s]["test_acc"] * 100 for s in stocks_tested]
bars1 = ax1.bar(
    stocks_tested, test_accs, color="steelblue", alpha=0.7, edgecolor="black"
)
ax1.axhline(y=50, color="red", linestyle="--", alpha=0.5, label="Random (50%)")
ax1.set_ylabel("Test Accuracy (%)", fontsize=11)
ax1.set_title("ML Model Test Accuracy by Stock", fontsize=12, fontweight="bold")
ax1.set_ylim([40, 70])
ax1.legend()
for bar in bars1:
    height = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{height:.1f}%",
        ha="center",
        va="bottom",
        fontweight="bold",
    )

# Plot 2: F1 Scores
ax2 = axes[0, 1]
f1_scores = [all_metrics[s]["f1"] * 100 for s in stocks_tested]
bars2 = ax2.bar(stocks_tested, f1_scores, color="green", alpha=0.7, edgecolor="black")
ax2.set_ylabel("F1 Score (%)", fontsize=11)
ax2.set_title("Model F1 Score by Stock", fontsize=12, fontweight="bold")
ax2.set_ylim([0, 100])
for bar in bars2:
    height = bar.get_height()
    ax2.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{height:.1f}%",
        ha="center",
        va="bottom",
        fontweight="bold",
    )

# Plot 3: Indicator Weights
ax3 = axes[1, 0]
indicators = list(
    sorted(normalized_weights.keys(), key=lambda x: normalized_weights[x], reverse=True)
)
weights = [normalized_weights[i] * 100 for i in indicators]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
bars3 = ax3.bar(
    indicators, weights, color=colors[: len(indicators)], alpha=0.7, edgecolor="black"
)
ax3.set_ylabel("Weight (%)", fontsize=11)
ax3.set_title("Normalized Indicator Weights (From ML)", fontsize=12, fontweight="bold")
for bar in bars3:
    height = bar.get_height()
    ax3.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{height:.1f}%",
        ha="center",
        va="bottom",
        fontweight="bold",
    )

# Plot 4: Feature Importance for First Stock
ax4 = axes[1, 1]
if stocks_tested:
    first_stock = stocks_tested[0]
    importance = all_feature_importance[first_stock].sort_values(ascending=True)
    ax4.barh(
        range(len(importance)),
        importance.values * 100,
        color="coral",
        alpha=0.7,
        edgecolor="black",
    )
    ax4.set_yticks(range(len(importance)))
    ax4.set_yticklabels(importance.index)
    ax4.set_xlabel("Importance (%)", fontsize=11)
    ax4.set_title(
        f"Feature Importance - {first_stock} (ML Model)", fontsize=12, fontweight="bold"
    )

plt.tight_layout()
plt.savefig(
output_dir/"images"/"ml_backtest_results.png", dpi=300, bbox_inches="tight"
)
print("\n✅ ML results visualization saved to: ml_backtest_results.png")
plt.show()

# ==================== Save Results ====================

results_config = {
    "timestamp": str(pd.Timestamp.now()),
    "method": "Random Forest Classifier (Time Series Split: 80-20)",
    "algorithm": "RandomForestClassifier(n_estimators=100, max_depth=10)",
    "target": "Binary classification - predict if price goes up or down in next period",
    "model_performance": {
        s: {
            "test_accuracy": round(all_metrics[s]["test_acc"], 4),
            "f1_score": round(all_metrics[s]["f1"], 4),
            "auc": round(all_metrics[s]["auc"], 4),
            "test_samples": all_metrics[s]["n_test_samples"],
        }
        for s in stocks_tested
    },
    "normalized_weights": {k: round(v, 4) for k, v in normalized_weights.items()},
    "notes": "Weights based on ML feature importance.Use these in formula",
}

output_path = output_dir / "ml_backtest_results.json"
with open(output_path, "w") as f:
    json.dump(results_config, f, indent=2)

print(f"\n✅ ML results saved to: {output_path}")

# ==================== Summary ====================

print("\n" + "=" * 80)
print("FINAL ML-BASED INDICATOR WEIGHTS FOR STEP 2")
print("=" * 80)

print("\nPython Code for Updated Scoring Formula:")
print(
    """
# ML-optimized weights (based on Random Forest feature importance)
ml_weights = {
"""
)
for indicator, weight in sorted(
    normalized_weights.items(), key=lambda x: x[1], reverse=True
):
    print(f"    '{indicator}': {weight:.4f},")
print(
    """}

composite_score = (
    rsi_score * ml_weights['RSI'] +
    macd_score * ml_weights['MACD'] +
    bb_score * ml_weights['BB'] +
    supertrend_score * ml_weights['Supertrend']
)
"""
)

print("\n" + "=" * 80)
print("✅ STEP 1 COMPLETE: ML-based indicator weights calculated")
print("📝 Next: Use these weights in scoring formula (Step 2)")
print("=" * 80)
