"""
USD/JPY Indicator Denoising via Rolling PCA
============================================
Identifies high-probability mean-reversion signals by applying Rolling PCA
to a 20+ indicator feature matrix built from 5 years of daily OHLCV data.

Usage:
    python usdjpy_rolling_pca.py
"""

import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─────────────────────────── configuration ────────────────────────────────────
TICKER = "USDJPY=X"
PERIOD = "5y"
PCA_WINDOW = 63          # ~3 trading months
N_COMPONENTS = 3
OUTPUT_PARQUET = "usdjpy_denoised.parquet"
FIGURE_FILE = "usdjpy_rolling_pca_dashboard.png"

# ──────────────────────────── Step 1: Data Ingestion ──────────────────────────

def _synthetic_usdjpy(n_days: int = 1260) -> pd.DataFrame:
    """
    Generate a realistic synthetic USD/JPY daily OHLCV series when the
    network is unavailable.  Used only as a fallback; real data takes
    priority when yfinance can reach the internet.
    """
    rng = np.random.default_rng(42)
    end   = pd.Timestamp("today").normalize()
    dates = pd.bdate_range(end=end, periods=n_days)

    # Geometric Brownian Motion with mild mean-reversion
    mu, sigma = 0.0001, 0.005
    log_returns = rng.normal(mu - 0.5 * sigma**2, sigma, n_days)
    # Add a slow mean-reverting component (Ornstein–Uhlenbeck flavour)
    ou = np.zeros(n_days)
    for i in range(1, n_days):
        ou[i] = 0.98 * ou[i - 1] + rng.normal(0, 0.003)
    log_returns += ou

    close = 130.0 * np.exp(np.cumsum(log_returns))      # start ~130 JPY/USD

    daily_range = close * rng.uniform(0.002, 0.008, n_days)
    high   = close + daily_range * rng.uniform(0.3, 0.7, n_days)
    low    = close - daily_range * rng.uniform(0.3, 0.7, n_days)
    open_  = low + (high - low) * rng.uniform(0, 1, n_days)
    volume = rng.integers(50_000, 200_000, n_days).astype(float)

    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )
    df.index.name = "Date"
    return df


def download_data(ticker: str = TICKER, period: str = PERIOD) -> pd.DataFrame:
    """
    Download OHLCV data via yfinance and validate / interpolate missing values.
    Falls back to a synthetic dataset if the network is unavailable.
    """
    print(f"[1] Downloading {ticker} ({period}) …")
    raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)

    if raw.empty:
        print("   ⚠  Network unavailable – falling back to synthetic USD/JPY data.")
        raw = _synthetic_usdjpy()
    else:
        # Flatten multi-level columns that yfinance sometimes returns
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        # Keep standard OHLCV columns
        raw = raw[["Open", "High", "Low", "Close", "Volume"]].copy()

    # Report and fill missing values
    missing = raw.isnull().sum().sum()
    if missing:
        print(f"   → {missing} missing value(s) detected – applying linear interpolation.")
        raw = raw.interpolate(method="linear").ffill().bfill()

    print(f"   → {len(raw)} trading days loaded  ({raw.index[0].date()} – {raw.index[-1].date()}).")
    return raw


# ──────────────────────── Step 2: Feature Engineering ─────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 20+ technical indicators and return a merged DataFrame."""
    print("[2] Building feature matrix …")

    o, h, l, c, v = df["Open"], df["High"], df["Low"], df["Close"], df["Volume"]

    feat = pd.DataFrame(index=df.index)

    # ── Momentum ──────────────────────────────────────────────────────────────
    rsi = ta.rsi(c, length=14)
    feat["RSI_14"] = rsi

    stoch = ta.stoch(h, l, c, k=14, d=3)
    feat["STOCH_K"] = stoch["STOCHk_14_3_3"]
    feat["STOCH_D"] = stoch["STOCHd_14_3_3"]

    feat["WILLR_14"] = ta.willr(h, l, c, length=14)
    feat["ROC_10"]   = ta.roc(c, length=10)
    feat["ROC_20"]   = ta.roc(c, length=20)

    # ── Trend ─────────────────────────────────────────────────────────────────
    macd = ta.macd(c, fast=12, slow=26, signal=9)
    feat["MACD"]        = macd["MACD_12_26_9"]
    feat["MACD_SIGNAL"] = macd["MACDs_12_26_9"]
    feat["MACD_HIST"]   = macd["MACDh_12_26_9"]

    adx = ta.adx(h, l, c, length=14)
    feat["ADX_14"]  = adx["ADX_14"]
    feat["DMP_14"]  = adx["DMP_14"]
    feat["DMN_14"]  = adx["DMN_14"]

    feat["EMA_9"]   = ta.ema(c, length=9)
    feat["EMA_21"]  = ta.ema(c, length=21)
    feat["EMA_50"]  = ta.ema(c, length=50)
    feat["EMA_200"] = ta.ema(c, length=200)

    # Express EMAs relative to price (ratio) to keep them stationary
    for span in [9, 21, 50, 200]:
        feat[f"EMA_{span}"] = c / feat[f"EMA_{span}"] - 1

    # ── Volatility ────────────────────────────────────────────────────────────
    bb = ta.bbands(c, length=20, std=2)
    feat["BB_UPPER"] = (bb["BBU_20_2.0_2.0"] - c) / c  # distance to upper band
    feat["BB_LOWER"] = (c - bb["BBL_20_2.0_2.0"]) / c  # distance to lower band
    feat["BB_PCT"]   = bb["BBP_20_2.0_2.0"]             # %B

    feat["ATR_14"]   = ta.atr(h, l, c, length=14) / c  # normalised ATR

    kc = ta.kc(h, l, c, length=20, scalar=1.5)
    feat["KC_UPPER"] = (kc["KCUe_20_1.5"] - c) / c
    feat["KC_LOWER"] = (c - kc["KCLe_20_1.5"]) / c

    # ── Merge with price ──────────────────────────────────────────────────────
    feat["Close"] = c.values

    feat.dropna(inplace=True)
    print(f"   → Feature matrix: {feat.shape[0]} rows × {feat.shape[1]} columns.")
    return feat


# ────────────────── Step 3: Pre-processing & Rolling Standardisation ──────────

def rolling_scale(X: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Apply a causal (no look-ahead) rolling z-score to each column.

    For row t:  z_t = (x_t - mean[t-window+1 : t]) / std[t-window+1 : t]
    Only rows where a full window is available are kept.
    """
    means = X.rolling(window=window, min_periods=window).mean()
    stds  = X.rolling(window=window, min_periods=window).std(ddof=1)
    scaled = (X - means) / stds.replace(0, np.nan)
    return scaled.dropna()


# ─────────────────────── Step 4: Rolling PCA ──────────────────────────────────

def rolling_pca(
    scaled: pd.DataFrame,
    feature_cols: list,
    window: int = PCA_WINDOW,
    n_components: int = N_COMPONENTS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fit PCA over a rolling window and return:
        pc_df       – DataFrame of PC scores (PC1, PC2, PC3) aligned to the
                      right edge of each window.
        loadings_df – PC1 loadings over time (feature × date).
        var_df      – Explained variance ratio for PC1–PC3 over time.
    """
    print(f"[4] Running Rolling PCA  (window={window}, components={n_components}) …")

    X = scaled[feature_cols].values
    dates = scaled.index

    pc_records       = []
    loading_records  = []
    var_records      = []

    n_rows = len(X)
    pca = PCA(n_components=n_components, svd_solver="full")
    for end in range(window - 1, n_rows):
        start = end - window + 1
        window_data = X[start : end + 1]          # shape (window, n_features)

        pca.fit(window_data)
        scores = pca.transform(window_data)       # (window, n_components)

        date = dates[end]

        # Only store the score at the last (current) date
        pc_records.append({
            "Date": date,
            "PC1": scores[-1, 0],
            "PC2": scores[-1, 1],
            "PC3": scores[-1, 2],
        })

        # PC1 component loadings
        loading_records.append(
            pd.Series(pca.components_[0], index=feature_cols, name=date)
        )

        # Explained variance ratios
        var_records.append({
            "Date": date,
            "EVR_PC1": pca.explained_variance_ratio_[0],
            "EVR_PC2": pca.explained_variance_ratio_[1],
            "EVR_PC3": pca.explained_variance_ratio_[2],
            "EVR_Total": pca.explained_variance_ratio_.sum(),
        })

    pc_df       = pd.DataFrame(pc_records).set_index("Date")
    loadings_df = pd.DataFrame(loading_records)           # dates × features
    var_df      = pd.DataFrame(var_records).set_index("Date")

    print(f"   → PC scores computed for {len(pc_df)} dates.")
    return pc_df, loadings_df, var_df


# ─────────────────────── Step 5: Visual Analytics ─────────────────────────────

def build_dashboard(
    price: pd.Series,
    pc_df: pd.DataFrame,
    loadings_df: pd.DataFrame,
    var_df: pd.DataFrame,
    feature_cols: list,
    raw_features: pd.DataFrame,
    save_path: str = FIGURE_FILE,
) -> None:
    """Produce a 4-panel analytical dashboard and save it to disk."""
    print("[5] Building dashboard …")

    fig = plt.figure(figsize=(20, 14), constrained_layout=True)
    fig.suptitle("USD/JPY – Indicator Denoising via Rolling PCA", fontsize=16, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig)

    # ── Panel 1: Price vs. PC1 (Denoised Momentum) ───────────────────────────
    ax1a = fig.add_subplot(gs[0, 0])
    ax1b = ax1a.twinx()

    aligned_price = price.reindex(pc_df.index)
    ax1a.plot(aligned_price.index, aligned_price.values, color="#1f77b4", linewidth=1.2,
              label="USD/JPY Close")
    ax1b.plot(pc_df.index, pc_df["PC1"], color="#ff7f0e", linewidth=1.0, alpha=0.85,
              label="PC1 (Denoised Momentum)")
    ax1b.axhline(0, color="grey", linestyle="--", linewidth=0.7)

    ax1a.set_title("USD/JPY Price vs. Denoised PC1 Oscillator")
    ax1a.set_ylabel("Price (JPY per USD)", color="#1f77b4")
    ax1b.set_ylabel("PC1 Score", color="#ff7f0e")
    ax1a.tick_params(axis="x", rotation=30)

    lines_a, labels_a = ax1a.get_legend_handles_labels()
    lines_b, labels_b = ax1b.get_legend_handles_labels()
    ax1a.legend(lines_a + lines_b, labels_a + labels_b, loc="upper left", fontsize=8)

    # ── Panel 2: PC1 Loading Heatmap ──────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])

    # Downsample to at most 50 date labels for readability
    step = max(1, len(loadings_df) // 50)
    hm_data = loadings_df.iloc[::step].T          # features × sampled_dates

    # Shorten date labels
    date_labels = [d.strftime("%Y-%m") if hasattr(d, "strftime") else str(d)
                   for d in hm_data.columns]

    sns.heatmap(
        hm_data,
        ax=ax2,
        cmap="RdBu_r",
        center=0,
        xticklabels=date_labels,
        yticklabels=feature_cols,
        cbar_kws={"shrink": 0.7},
        linewidths=0,
    )
    ax2.set_title("PC1 Feature Loadings Over Time")
    ax2.set_xlabel("Date")
    ax2.tick_params(axis="x", rotation=45, labelsize=6)
    ax2.tick_params(axis="y", labelsize=7)

    # ── Panel 3: Explained Variance Ratio ────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(var_df.index, var_df["EVR_PC1"] * 100, label="PC1", linewidth=1.2)
    ax3.plot(var_df.index, var_df["EVR_PC2"] * 100, label="PC2", linewidth=1.0, alpha=0.8)
    ax3.plot(var_df.index, var_df["EVR_PC3"] * 100, label="PC3", linewidth=1.0, alpha=0.8)
    ax3.plot(var_df.index, var_df["EVR_Total"] * 100, label="PC1+2+3",
             linewidth=1.2, linestyle="--", color="black")
    ax3.set_title("Rolling Explained Variance Ratio (PC1–PC3)")
    ax3.set_ylabel("Variance Explained (%)")
    ax3.legend(fontsize=8)
    ax3.tick_params(axis="x", rotation=30)
    ax3.grid(True, alpha=0.3)

    # ── Panel 4: Correlation Matrix of Raw Indicators ─────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    corr = raw_features[feature_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        ax=ax4,
        mask=mask,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        annot=False,
        linewidths=0,
        cbar_kws={"shrink": 0.7},
        xticklabels=feature_cols,
        yticklabels=feature_cols,
    )
    ax4.set_title("Indicator Correlation Matrix (Raw Features)")
    ax4.tick_params(axis="x", rotation=90, labelsize=6)
    ax4.tick_params(axis="y", labelsize=6)

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"   → Dashboard saved → {save_path}")
    plt.close(fig)


# ──────────────────────────── Output: Parquet ─────────────────────────────────

def save_parquet(
    price: pd.Series,
    pc_df: pd.DataFrame,
    var_df: pd.DataFrame,
    path: str = OUTPUT_PARQUET,
) -> None:
    """Combine price + PC scores + variance info and write to Parquet."""
    out = pd.concat(
        [price.rename("Close"), pc_df, var_df],
        axis=1,
        join="inner",
    )
    out.index.name = "Date"
    out.to_parquet(path, index=True)
    print(f"[6] Denoised dataset saved → {path}  ({out.shape[0]} rows × {out.shape[1]} cols).")


# ─────────────────────────────── main ─────────────────────────────────────────

def main() -> None:
    # 1. Data ingestion
    raw = download_data()

    # 2. Feature engineering
    features = build_features(raw)
    feature_cols = [c for c in features.columns if c != "Close"]

    # 3. Rolling standardisation (causal – no look-ahead bias)
    print("[3] Applying rolling standardisation …")
    scaled = rolling_scale(features[feature_cols], window=PCA_WINDOW)
    print(f"   → Scaled matrix: {scaled.shape[0]} rows × {scaled.shape[1]} columns.")

    # 4. Rolling PCA
    pc_df, loadings_df, var_df = rolling_pca(scaled, feature_cols)

    # 5. Visual analytics
    build_dashboard(
        price=features["Close"],
        pc_df=pc_df,
        loadings_df=loadings_df,
        var_df=var_df,
        feature_cols=feature_cols,
        raw_features=features,
    )

    # 6. Save to Parquet
    save_parquet(features["Close"], pc_df, var_df)

    print("\n✓ Pipeline complete.")


if __name__ == "__main__":
    main()
