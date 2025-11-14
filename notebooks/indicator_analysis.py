"""
indicator_analysis.py
ENHANCED Indicator Analysis with ML-Weighted Scores (Technical + Fundamental)

- Loads ML-optimized weights from backtest_ml_model.py
- Calculates technical + fundamental scores for each stock
- Applies ML weights to get final composite score
- Generates visualizations and signals
- Outputs stock rankings and trading signals

Run: python notebooks/indicator_analysis.py
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
import os

warnings.filterwarnings("ignore")

# Add src to path (works in both terminal and Jupyter)
current_dir = os.getcwd()
src_path = os.path.join(current_dir, "src")
if os.path.exists(src_path):
    sys.path.insert(0, src_path)

from nse_analyzer.utils.config import ConfigManager

# ==================== Setup ====================

config_manager = ConfigManager()
data_dir = config_manager.get_data_dir()
indicators_dir = data_dir / "indicators"
fundamentals_dir = data_dir / "fundamentals"

# For notebooks: use relative path
try:
    output_dir = Path("notebooks")
except:
    output_dir = Path.cwd() / "notebooks"

output_dir.mkdir(parents=True, exist_ok=True)
(output_dir / "images").mkdir(exist_ok=True)

stocks = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "KOTAKBANK"]

print("=" * 80)
print("ML BACKTEST v3 - TECHNICAL + FUNDAMENTAL INDICATORS")
print("=" * 80)

# ==================== Load ML Weights ====================

def load_ml_weights():
    """Load ML-optimized weights from backtest results."""
    
    # Try multiple possible locations
    possible_paths = [
        Path("notebooks/ml_backtest_results.json"),
        Path.cwd() / "notebooks" / "ml_backtest_results.json",
        Path(__file__).parent/ "ml_backtest_results.json",
        Path.cwd() / "ml_backtest_results.json",
    ]
    
    weights_file = None
    for path in possible_paths:
        if path.exists():
            weights_file = path
            print(f"✅ Found ML weights at: {path}")
            break
    
    if weights_file is None:
        print(f"⚠️  ML weights file not found in any of these locations:")
        for path in possible_paths:
            print(f"   - {path}")
        print("\n   Running basic analysis with fixed weights.")
        return {
            "RSI": 0.25,
            "MACD": 0.25,
            "BB": 0.25,
            "Supertrend": 0.25,
            "FUNDAMENTAL": 0.00,
        }
    
    try:
        with open(weights_file, "r") as f:
            results = json.load(f)
        return results.get("normalized_weights", {})
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return {
            "RSI": 0.25,
            "MACD": 0.25,
            "BB": 0.25,
            "Supertrend": 0.25,
            "FUNDAMENTAL": 0.00,
        }

ml_weights = load_ml_weights()

print(f"\n✅ Loaded ML Weights:")
for indicator, weight in sorted(ml_weights.items(), key=lambda x: x[1], reverse=True):
    print(f"   {indicator:20} | {weight*100:6.1f}%")

# ==================== Load Data ====================

def load_indicator_data(stock_name):
    """Load technical indicators."""
    csv_path = indicators_dir / f"{stock_name}_indicators.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df["Date"] = pd.to_datetime(df["Date"])
        return df.sort_values("Date").reset_index(drop=True)
    return None

def load_fundamental_data():
    """Load fundamental indicators."""
    csv_path = fundamentals_dir / "fundamentals.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df['ticker'] = df['ticker'].str.replace('.NS', '')
        return df.set_index('ticker')
    return None

# Load all data
technical_data = {}
for stock in stocks:
    df = load_indicator_data(stock)
    if df is not None:
        technical_data[stock] = df

fundamental_data = load_fundamental_data()

print(f"\n✅ Loaded technical indicators for {len(technical_data)} stocks")
if fundamental_data is not None:
    print(f"✅ Loaded fundamental data for {len(fundamental_data)} stocks")

# ==================== Calculate Scores ====================

def calculate_technical_score(df):
    """Calculate technical score from latest indicators."""
    if df is None or len(df) == 0:
        return None
    
    latest = df.iloc[-1]
    
    # RSI Score (0-100)
    rsi = latest.get('RSI_14', 50)
    if pd.isna(rsi):
        rsi_score = 50
    else:
        rsi_score = rsi  # Already 0-100
    
    # MACD Score (0-100)
    macd = latest.get('MACD', 0)
    macd_signal = latest.get('MACD_Signal', 0)
    if pd.isna(macd) or pd.isna(macd_signal):
        macd_score = 50
    else:
        macd_diff = macd - macd_signal
        macd_score = 50 + (np.tanh(macd_diff * 10) * 50)
        macd_score = max(0, min(100, macd_score))
    
    # Bollinger Bands Score (0-100)
    upper = latest.get('BB_Upper_20')
    lower = latest.get('BB_Lower_20')
    close = latest.get('Close', 0)
    
    if pd.isna(upper) or pd.isna(lower) or pd.isna(close):
        bb_score = 50
    else:
        if (upper - lower) > 0:
            bb_percent = (close - lower) / (upper - lower)
        else:
            bb_percent = 0.5
        bb_score = int(bb_percent * 100)
        bb_score = max(0, min(100, bb_score))
    
    # Supertrend Score (0-100)
    st_direction = latest.get('Supertrend_Direction', 0)
    if pd.isna(st_direction):
        supertrend_score = 50
    else:
        supertrend_score = 70 if st_direction > 0 else 30
    
    return {
        'RSI': rsi_score,
        'MACD': macd_score,
        'BB': bb_score,
        'Supertrend': supertrend_score
    }

def calculate_fundamental_score(stock_name, fundamental_data_df):
    """Calculate composite fundamental score (0-100)."""
    if fundamental_data_df is None or stock_name not in fundamental_data_df.index:
        return 50  # Neutral if no data
    
    fund = fundamental_data_df.loc[stock_name]
    score = 50  # Start neutral
    
    # PE Ratio scoring (lower is better, but not too low)
    pe = fund.get('pe_ratio')
    if pd.notna(pe) and pe > 0:
        if 15 <= pe <= 30:
            score += 10
        elif pe < 15:
            score += 5
        elif pe > 30:
            score -= 5
    
    # ROE scoring (higher is better)
    roe = fund.get('roe')
    if pd.notna(roe):
        roe_pct = roe * 100 if roe < 1 else roe
        if roe_pct > 15:
            score += 15
        elif roe_pct > 10:
            score += 10
        elif roe_pct > 5:
            score += 5
    
    # Debt-to-Equity scoring (lower is better)
    de = fund.get('debt_to_equity')
    if pd.notna(de) and de > 0:
        if de < 1:
            score += 10
        elif de < 2:
            score += 5
        else:
            score -= 5
    
    # Dividend Yield scoring
    dy = fund.get('dividend_yield')
    if pd.notna(dy) and dy > 0:
        dy_pct = dy * 100 if dy < 1 else dy
        if dy_pct > 2:
            score += 5
    
    # Revenue Growth scoring
    rg = fund.get('revenue_growth_yoy')
    if pd.notna(rg):
        if rg > 10:
            score += 10
        elif rg > 5:
            score += 5
    
    # Net Profit Margin scoring
    npm = fund.get('net_profit_margin')
    if pd.notna(npm):
        npm_pct = npm * 100 if npm < 1 else npm
        if npm_pct > 15:
            score += 10
        elif npm_pct > 10:
            score += 5
    
    return min(100, max(0, score))

# ==================== Generate Stock Analysis ====================

print("\n" + "=" * 80)
print("STEP 1: CALCULATING SCORES FOR ALL STOCKS")
print("=" * 80)

results = []

for stock in stocks:
    print(f"\n{stock}:")
    
    if stock not in technical_data:
        print("  ❌ No technical data")
        continue
    
    # Technical scores
    tech_scores = calculate_technical_score(technical_data[stock])
    if tech_scores is None:
        print("  ⚠️ Cannot calculate technical scores")
        continue
    
    print(f"  Technical Scores:")
    for indicator, score in tech_scores.items():
        print(f"    {indicator:15} | {score:6.1f}")
    
    # Fundamental score
    fund_score = calculate_fundamental_score(stock, fundamental_data)
    print(f"  Fundamental Score: {fund_score:6.1f}")
    
    # Weighted composite score
    composite = (
        tech_scores['RSI'] * ml_weights.get('RSI', 0.25) +
        tech_scores['MACD'] * ml_weights.get('MACD', 0.25) +
        tech_scores['BB'] * ml_weights.get('BB', 0.25) +
        tech_scores['Supertrend'] * ml_weights.get('Supertrend', 0.25) +
        fund_score * ml_weights.get('FUNDAMENTAL', 0)
    )
    
    # Determine signal
    if composite >= 65:
        signal = "🟢 STRONG BUY"
    elif composite >= 55:
        signal = "🟢 BUY"
    elif composite >= 45:
        signal = "🟡 HOLD"
    elif composite >= 35:
        signal = "🔴 SELL"
    else:
        signal = "🔴 STRONG SELL"
    
    print(f"  Composite Score: {composite:6.1f}")
    print(f"  Signal: {signal}")
    
    results.append({
        'Stock': stock,
        'RSI': tech_scores['RSI'],
        'MACD': tech_scores['MACD'],
        'BB': tech_scores['BB'],
        'Supertrend': tech_scores['Supertrend'],
        'Fundamental': fund_score,
        'Composite': composite,
        'Signal': signal,
    })

results_df = pd.DataFrame(results).sort_values('Composite', ascending=False)

# ==================== Visualizations ====================

print("\n" + "=" * 80)
print("STEP 2: GENERATING VISUALIZATIONS")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Heatmap of all scores
ax1 = axes[0, 0]
heatmap_data = results_df[['Stock', 'RSI', 'MACD', 'BB', 'Supertrend', 'Fundamental', 'Composite']].set_index('Stock')
im1 = ax1.imshow(heatmap_data.T, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
ax1.set_yticks(range(len(heatmap_data.columns)))
ax1.set_yticklabels(heatmap_data.columns)
ax1.set_xticks(range(len(heatmap_data)))
ax1.set_xticklabels(heatmap_data.index)
ax1.set_title('Indicator Scores Heatmap (0-100)', fontsize=12, fontweight='bold')
plt.colorbar(im1, ax=ax1, label='Score')

# Add values to heatmap
for i in range(len(heatmap_data.columns)):
    for j in range(len(heatmap_data)):
        text = ax1.text(j, i, f'{heatmap_data.iloc[j, i]:.0f}',
                       ha="center", va="center", color="black", fontweight='bold', fontsize=9)

# Plot 2: Composite scores with ranking
ax2 = axes[0, 1]
colors = ['green' if s.startswith('🟢 STRONG') else 'lightgreen' if s.startswith('🟢') 
          else 'yellow' if s.startswith('🟡') else 'salmon' if s.startswith('🔴 SELL') else 'red' 
          for s in results_df['Signal']]
bars = ax2.barh(results_df['Stock'], results_df['Composite'], color=colors, edgecolor='black', linewidth=1.5)
ax2.axvline(x=50, color='black', linestyle='--', alpha=0.5, label='Neutral (50)')
ax2.set_xlabel('Composite Score', fontsize=11, fontweight='bold')
ax2.set_title('Final Composite Scores (ML-Weighted)', fontsize=12, fontweight='bold')
ax2.set_xlim([0, 100])
ax2.legend()
for i, (idx, row) in enumerate(results_df.iterrows()):
    ax2.text(row['Composite'] + 1, i, f"{row['Composite']:.1f}", va='center', fontweight='bold')

# Plot 3: Technical vs Fundamental breakdown
ax3 = axes[1, 0]
x_pos = np.arange(len(results_df))
width = 0.35

tech_avg = results_df[['RSI', 'MACD', 'BB', 'Supertrend']].mean(axis=1)
bars1 = ax3.bar(x_pos - width/2, tech_avg, width, label='Technical Avg', alpha=0.8, color='steelblue')
bars2 = ax3.bar(x_pos + width/2, results_df['Fundamental'], width, label='Fundamental', alpha=0.8, color='coral')

ax3.set_ylabel('Score (0-100)', fontsize=11, fontweight='bold')
ax3.set_title('Technical vs Fundamental Scores', fontsize=12, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(results_df['Stock'])
ax3.legend()
ax3.set_ylim([0, 100])

# Plot 4: Individual indicator contributions
ax4 = axes[1, 1]
indicators = ['RSI', 'MACD', 'BB', 'Supertrend', 'Fundamental']
colors_ind = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#d62728']

for idx, stock in enumerate(results_df['Stock']):
    values = [results_df.iloc[idx][ind] for ind in indicators]
    angles = np.linspace(0, 2 * np.pi, len(indicators), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    
    ax4 = plt.subplot(2, 2, 4, projection='polar')
    ax4.plot(angles, values, 'o-', linewidth=2, label=stock)
    ax4.fill(angles, values, alpha=0.15)

ax4.set_xticks(np.linspace(0, 2 * np.pi, len(indicators), endpoint=False))
ax4.set_xticklabels(indicators)
ax4.set_ylim(0, 100)
ax4.set_title('Indicator Profile (Radar Chart)', fontsize=12, fontweight='bold', pad=20)
ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax4.grid(True)

plt.tight_layout()
plt.savefig(output_dir / "images" / "indicator_analysis_technical_fundamental.png", dpi=300, bbox_inches="tight")
print(f"✅ Visualization saved to: {output_dir / 'images' / 'indicator_analysis_technical_fundamental.png'}")
plt.show()

# ==================== Summary Report ====================

print("\n" + "=" * 80)
print("STOCK RANKING & SIGNALS")
print("=" * 80)

print("\n" + results_df[['Stock', 'RSI', 'MACD', 'BB', 'Supertrend', 'Fundamental', 'Composite', 'Signal']].to_string(index=False))

# Save results
results_csv = output_dir / "stock_analysis_results.csv"
results_df.to_csv(results_csv, index=False)
print(f"\n✅ Results saved to: {results_csv}")

# Save detailed analysis
analysis_json = {
    "timestamp": datetime.now().isoformat(),
    "ml_weights": ml_weights,
    "stocks_analysis": results_df.to_dict('records'),
    "recommendations": {
        "buy": results_df[results_df['Signal'].str.contains('BUY')]['Stock'].tolist(),
        "hold": results_df[results_df['Signal'].str.contains('HOLD')]['Stock'].tolist(),
        "sell": results_df[results_df['Signal'].str.contains('SELL')]['Stock'].tolist(),
    }
}

analysis_file = output_dir / "stock_analysis_detailed.json"
with open(analysis_file, "w") as f:
    json.dump(analysis_json, f, indent=2)

print(f"✅ Detailed analysis saved to: {analysis_file}")

print("\n" + "=" * 80)
print("✅ INDICATOR ANALYSIS COMPLETE")
print("=" * 80)
