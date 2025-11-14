"""
backtest_ml_model_v3.py
ENHANCED Machine Learning Backtesting Model with Fundamental Indicators

- Integrates 4 technical indicators + fundamental metrics
- Fundamental indicators weighted LOW (5-10%) as they have long-term impact
- Trains Random Forest on combined technical + fundamental features
- Tests on out-of-sample data
- Calculates feature importance (indicator weights)
- Reports actual ML model accuracy with visualization

Run: python notebooks/backtest_ml_model.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
import json
from datetime import datetime

warnings.filterwarnings("ignore")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nse_analyzer.utils.config import ConfigManager

# ==================== Setup ====================

config_manager = ConfigManager()
data_dir = config_manager.get_data_dir()
indicators_dir = data_dir / "indicators"
fundamentals_dir = data_dir / "fundamentals"
output_dir = Path(__file__).parent / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)
(output_dir / "images").mkdir(exist_ok=True)

stocks = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "KOTAKBANK"]

print("=" * 80)
print("ML BACKTEST v3 - TECHNICAL + FUNDAMENTAL INDICATORS")
print("=" * 80)

# ==================== Load Fundamental Data ====================

def load_fundamental_data():
    """Load fundamental indicators for all stocks."""
    fundamentals_csv = fundamentals_dir / "fundamentals.csv"
    
    if not fundamentals_csv.exists():
        print(f"⚠️  Fundamental data not found at {fundamentals_csv}")
        print("   Run: python -m src.nse_analyzer.indicators.fundamental")
        return None
    
    df = pd.read_csv(fundamentals_csv)
    df['ticker'] = df['ticker'].str.replace('.NS', '')
    return df.set_index('ticker')

fundamental_data = load_fundamental_data()

if fundamental_data is not None:
    print(f"✅ Loaded fundamental data for {len(fundamental_data)} stocks")
    print(f"   Columns: {list(fundamental_data.columns)[:5]}...")
else:
    print("❌ Fundamental data not available - using technical indicators only")
    fundamental_data = {}

# ==================== Load Technical Indicator Data ====================

def load_indicator_data(stock_name):
    """Load technical indicator CSV for a stock."""
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

def prepare_ml_data(stock_name, df, fundamental_data_dict, target_periods_ahead=1):
    """
    Prepare data for ML model
    
    Technical Features: RSI, MACD, BB, Supertrend
    Fundamental Features: PE, PB, ROE, Debt-to-Equity, etc. (low weight threshold)
    Target: 1 if price goes up in next period, 0 if goes down
    """
    df = df.copy()
    
    df["Future_Close"] = df["Close"].shift(-target_periods_ahead)
    df["Target"] = (df["Future_Close"] > df["Close"]).astype(int)
    
    technical_features = [
        "RSI_14", "MACD", "MACD_Signal", "MACD_Histogram",
        "Supertrend_Direction", "BB_Upper_20", "BB_Middle_20",
        "BB_Lower_20", "BB_Percent_B_20"
    ]
    
    fundamental_features = [
        "pe_ratio", "pb_ratio", "roe", "roce",
        "net_profit_margin", "debt_to_equity", "dividend_yield", "current_ratio"
    ]
    
    available_technical = [f for f in technical_features if f in df.columns]
    
    fundamental_values = {}
    if stock_name in fundamental_data_dict.index:
        stock_fundamentals = fundamental_data_dict.loc[stock_name]
        for feat in fundamental_features:
            if feat in stock_fundamentals.index and pd.notna(stock_fundamentals[feat]):
                fundamental_values[feat] = float(stock_fundamentals[feat])
    
    df_clean = df[available_technical + ["Target"]].dropna()
    
    if len(df_clean) == 0:
        return None, None, None, None
    
    X_technical = df_clean[available_technical]
    
    X_fundamental = pd.DataFrame(index=X_technical.index)
    for feat, value in fundamental_values.items():
        X_fundamental[feat] = value
    
    X = pd.concat([X_technical, X_fundamental], axis=1)
    
    y = df_clean["Target"]
    
    all_features = available_technical + list(fundamental_values.keys())
    
    return X, y, all_features, available_technical + list(fundamental_values.keys())

# ==================== Train ML Models ====================

print("\n" + "=" * 80)
print("TRAINING RANDOM FOREST MODELS (Technical + Fundamental)")
print("=" * 80)

all_feature_importance = {}
all_metrics = {}
feature_to_type = {}

for stock in stocks:
    print(f"\n{stock}:")
    
    if stock not in data:
        print(" ❌ No data available")
        continue
    
    X, y, features, all_features = prepare_ml_data(stock, data[stock], fundamental_data)
    
    if X is None or len(X) < 50:
        print(f" ⚠️ Insufficient data ({len(X) if X is not None else 0} samples)")
        continue
    
    n_technical = len([f for f in all_features if f.startswith(('RSI', 'MACD', 'BB', 'Supertrend'))])
    n_fundamental = len(all_features) - n_technical
    print(f" Features: {len(all_features)} ({n_technical} technical, {n_fundamental} fundamental)")
    
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    rf = RandomForestClassifier(
        n_estimators=150, max_depth=12, random_state=42, n_jobs=-1, class_weight='balanced'
    )
    rf.fit(X_train_scaled, y_train)
    
    y_pred_train = rf.predict(X_train_scaled)
    y_pred_test = rf.predict(X_test_scaled)
    y_pred_proba = rf.predict_proba(X_test_scaled)[:, 1]
    
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test, zero_division=0)
    recall = recall_score(y_test, y_pred_test, zero_division=0)
    f1 = f1_score(y_test, y_pred_test, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f" Train Accuracy: {train_acc*100:.2f}%")
    print(f" Test Accuracy:  {test_acc*100:.2f}%")
    print(f" Precision:      {precision*100:.2f}%")
    print(f" Recall:         {recall*100:.2f}%")
    print(f" F1 Score:       {f1*100:.2f}%")
    print(f" AUC-ROC:        {auc*100:.2f}%")
    
    importance = pd.Series(rf.feature_importances_, index=all_features).sort_values(ascending=False)
    
    print("\n Feature Importance:")
    for feat, imp in importance.items():
        feat_type = "FUND" if not any(feat.startswith(p) for p in ['RSI', 'MACD', 'BB', 'Supertrend']) else "TECH"
        print(f" {feat:20} [{feat_type}] | {imp*100:6.2f}%")
        if feat not in feature_to_type:
            feature_to_type[feat] = feat_type
    
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

# ==================== Calculate Weighted Indicators ====================

print("\n" + "=" * 80)
print("FEATURE & INDICATOR WEIGHT CALCULATION")
print("=" * 80)

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
    "pe_ratio": "FUNDAMENTAL",
    "pb_ratio": "FUNDAMENTAL",
    "roe": "FUNDAMENTAL",
    "roce": "FUNDAMENTAL",
    "net_profit_margin": "FUNDAMENTAL",
    "debt_to_equity": "FUNDAMENTAL",
    "dividend_yield": "FUNDAMENTAL",
    "current_ratio": "FUNDAMENTAL",
}

indicator_importance = {}
for stock, importance in all_feature_importance.items():
    for feature, imp_value in importance.items():
        indicator = feature_to_indicator.get(feature, "Other")
        if indicator not in indicator_importance:
            indicator_importance[indicator] = []
        indicator_importance[indicator].append(imp_value)

avg_importance = {}
for indicator, values in indicator_importance.items():
    avg_importance[indicator] = np.mean(values)

technical_imp_sum = sum([v for k, v in avg_importance.items() if k != "FUNDAMENTAL"])
fundamental_imp = avg_importance.get("FUNDAMENTAL", 0)

max_fundamental_weight = 0.10
if fundamental_imp > 0:
    fundamental_imp = min(fundamental_imp, max_fundamental_weight)
    avg_importance["FUNDAMENTAL"] = fundamental_imp

total_imp = sum(avg_importance.values())
normalized_weights = {k: v / total_imp for k, v in avg_importance.items()}

if "FUNDAMENTAL" in normalized_weights:
    current_fund_weight = normalized_weights["FUNDAMENTAL"]
    if current_fund_weight > 0.10:
        excess = current_fund_weight - 0.10
        normalized_weights["FUNDAMENTAL"] = 0.10
        
        technical_indicators = [k for k in normalized_weights if k != "FUNDAMENTAL"]
        for ind in technical_indicators:
            normalized_weights[ind] += excess * (normalized_weights[ind] / (1 - current_fund_weight))

total_norm = sum(normalized_weights.values())
normalized_weights = {k: v / total_norm for k, v in normalized_weights.items()}

print("\nRaw Feature Importance by Indicator (Average across stocks):")
for indicator, weight in sorted(avg_importance.items(), key=lambda x: x[1], reverse=True):
    print(f" {indicator:20} | {weight*100:6.2f}%")

print("\nNormalized Weights (Sum = 100%, Fundamental capped at 10%):")
for indicator, weight in sorted(normalized_weights.items(), key=lambda x: x[1], reverse=True):
    print(f" {indicator:20} | {weight*100:6.2f}%")

# ==================== Visualizations ====================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

ax1 = axes[0, 0]
stocks_tested = [s for s in stocks if s in all_metrics]
test_accs = [all_metrics[s]["test_acc"] * 100 for s in stocks_tested]

bars1 = ax1.bar(stocks_tested, test_accs, color="steelblue", alpha=0.7, edgecolor="black", linewidth=1.5)
ax1.axhline(y=50, color="red", linestyle="--", alpha=0.5, label="Random (50%)")
ax1.set_ylabel("Test Accuracy (%)", fontsize=11, fontweight="bold")
ax1.set_title("ML Model Test Accuracy (Technical + Fundamental)", fontsize=12, fontweight="bold")
ax1.set_ylim([40, 75])
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.1f}%",
             ha="center", va="bottom", fontweight="bold", fontsize=10)

ax2 = axes[0, 1]
f1_scores = [all_metrics[s]["f1"] * 100 for s in stocks_tested]

bars2 = ax2.bar(stocks_tested, f1_scores, color="green", alpha=0.7, edgecolor="black", linewidth=1.5)
ax2.set_ylabel("F1 Score (%)", fontsize=11, fontweight="bold")
ax2.set_title("Model F1 Score by Stock", fontsize=12, fontweight="bold")
ax2.set_ylim([0, 100])
ax2.grid(axis='y', alpha=0.3)

for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.1f}%",
             ha="center", va="bottom", fontweight="bold", fontsize=10)

ax3 = axes[1, 0]
indicators = list(sorted(normalized_weights.keys(), key=lambda x: normalized_weights[x], reverse=True))
weights = [normalized_weights[i] * 100 for i in indicators]
colors_map = {"FUNDAMENTAL": "#d62728", "RSI": "#1f77b4", "MACD": "#ff7f0e", "BB": "#2ca02c", "Supertrend": "#9467bd"}
colors = [colors_map.get(ind, "#7f7f7f") for ind in indicators]

bars3 = ax3.bar(indicators, weights, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)
ax3.set_ylabel("Weight (%)", fontsize=11, fontweight="bold")
ax3.set_title("Normalized Indicator Weights (ML-Optimized)", fontsize=12, fontweight="bold")
ax3.set_ylim([0, max(weights) * 1.15])
ax3.grid(axis='y', alpha=0.3)

for bar in bars3:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.1f}%",
             ha="center", va="bottom", fontweight="bold", fontsize=10)

ax4 = axes[1, 1]
metrics_names = ["Test Acc", "Precision", "Recall", "F1", "AUC"]
metrics_data = {}

for stock in stocks_tested:
    metrics_data[stock] = [
        all_metrics[stock]["test_acc"] * 100,
        all_metrics[stock]["precision"] * 100,
        all_metrics[stock]["recall"] * 100,
        all_metrics[stock]["f1"] * 100,
        all_metrics[stock]["auc"] * 100,
    ]

x_pos = np.arange(len(metrics_names))
width = 0.15

for idx, stock in enumerate(stocks_tested):
    ax4.bar(x_pos + idx * width, metrics_data[stock], width, label=stock, alpha=0.8)

ax4.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
ax4.set_title("All ML Metrics by Stock", fontsize=12, fontweight="bold")
ax4.set_xticks(x_pos + width * (len(stocks_tested) - 1) / 2)
ax4.set_xticklabels(metrics_names)
ax4.legend(loc="lower right", fontsize=9)
ax4.set_ylim([0, 110])
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "images" / "ml_backtest_technical_fundamental.png", dpi=300, bbox_inches="tight")
print(f"\n✅ Visualization saved to: {output_dir / 'images' / 'ml_backtest_technical_fundamental.png'}")
plt.show()

# ==================== Save Results ====================

results_config = {
    "timestamp": datetime.now().isoformat(),
    "method": "Random Forest Classifier (Technical + Fundamental Features)",
    "algorithm": "RandomForestClassifier(n_estimators=150, max_depth=12, class_weight='balanced')",
    "target": "Binary classification - predict if price goes up or down in next period",
    "features": {
        "technical": ["RSI", "MACD", "BB", "Supertrend"],
        "fundamental": ["PE_Ratio", "PB_Ratio", "ROE", "ROCE", "Net_Profit_Margin", "Debt_to_Equity", "Dividend_Yield", "Current_Ratio"],
        "note": "Fundamental indicators weighted LOW (capped at 10%) as they have long-term impact, not short-term"
    },
    "model_performance": {
        s: {
            "test_accuracy": round(all_metrics[s]["test_acc"], 4),
            "precision": round(all_metrics[s]["precision"], 4),
            "recall": round(all_metrics[s]["recall"], 4),
            "f1_score": round(all_metrics[s]["f1"], 4),
            "auc": round(all_metrics[s]["auc"], 4),
            "test_samples": all_metrics[s]["n_test_samples"],
        }
        for s in stocks_tested
    },
    "normalized_weights": {k: round(v, 4) for k, v in normalized_weights.items()},
    "notes": "Use these weights in scoring formula. Fundamentals are constrained to NOT exceed 10% total weight.",
}

output_path = output_dir / "ml_backtest_results.json"
with open(output_path, "w") as f:
    json.dump(results_config, f, indent=2)

print(f"\n✅ ML results saved to: {output_path}")

# ==================== Final Output ====================

print("\n" + "=" * 80)
print("FINAL INDICATOR WEIGHTS FOR SCORING")
print("=" * 80)

print("\nPython Code (for use in indicator_analysis.py):")
print("\nml_weights = {")
for indicator, weight in sorted(normalized_weights.items(), key=lambda x: x[1], reverse=True):
    print(f"    '{indicator}': {weight:.4f},  # {weight*100:.1f}%")
print("}")

print("\n" + "=" * 80)
print("✅ BACKTEST COMPLETE: ML weights calculated (Technical + Fundamental integrated)")
print("📝 Next: Use these weights in indicator_analysis.py")
print("=" * 80)
