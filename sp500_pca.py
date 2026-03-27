"""
Principal Component Analysis (PCA) on S&P 500 Stocks
======================================================
This script:
  1. Downloads the current S&P 500 constituent list from Wikipedia.
  2. Fetches two years of daily adjusted-close prices via yfinance.
  3. Computes daily log-returns and drops tickers with excessive missing data.
  4. Standardises the return matrix and runs PCA.
  5. Produces four publication-quality charts saved to ./pca_output/:
       - scree_plot.png          – explained-variance ratio per component
       - cumulative_variance.png – cumulative explained variance
       - top_loadings.png        – top stock weights in the first 4 PCs
       - biplot.png              – scatter of PC1 vs PC2 scores with loadings

When network access is unavailable the script automatically falls back to a
bundled set of ~100 major S&P 500 tickers and generates synthetic (but
realistic) correlated return data so the full PCA pipeline can still be
demonstrated.
"""

import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────
OUTPUT_DIR = "pca_output"
N_COMPONENTS = 20          # number of PCs to compute and display
N_YEARS = 2                # historical window in years
MIN_DATA_FRACTION = 0.80   # drop tickers with < 80 % of trading days present
TOP_N_LOADINGS = 15        # stocks to show in the loadings chart
N_BIPLOT_ARROWS = 20       # loading vectors shown on the biplot
RANDOM_STATE = 42

sns.set_theme(style="whitegrid", palette="muted")

# ── Fallback ticker list (major S&P 500 constituents) ─────────────────────────
_FALLBACK_TICKERS: list[str] = [
    # Information Technology
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CSCO", "ACN", "IBM", "TXN", "AMAT",
    "QCOM", "NOW", "INTU", "AMD", "MU", "LRCX", "ADI", "KLAC", "MCHP", "HPQ",
    # Health Care
    "UNH", "LLY", "JNJ", "ABBV", "MRK", "TMO", "ABT", "DHR", "PFE", "AMGN",
    "ISRG", "SYK", "BMY", "GILD", "BSX", "VRTX", "MDT", "HCA", "CI", "ELV",
    # Financials
    "BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "BLK", "SPGI",
    "AXP", "CB", "MMC", "AON", "TRV", "USB", "PGR", "MET", "AIG", "PRU",
    # Consumer Discretionary
    "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG", "CMG",
    "F", "GM", "ORLY", "AZO", "DHI", "LEN", "PHM", "ROST", "YUM", "HLT",
    # Communication Services
    "META", "GOOGL", "GOOG", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS", "EA",
    # Consumer Staples
    "PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "CL", "MDLZ", "GIS",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "PXD", "MPC", "PSX", "VLO", "OXY",
    # Industrials
    "GE", "HON", "CAT", "UPS", "DE", "RTX", "LMT", "BA", "MMM", "ITW",
    # Materials
    "LIN", "APD", "ECL", "NEM", "FCX", "NUE", "VMC", "MLM", "ALB", "CF",
    # Utilities
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL", "PCG", "ED",
]


# ── Helper functions ───────────────────────────────────────────────────────────

def fetch_sp500_tickers() -> list[str]:
    """Return S&P 500 ticker symbols.

    Attempts to scrape the live Wikipedia constituent table.  Falls back to the
    bundled list when network access is unavailable.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        tables = pd.read_html(url, attrs={"id": "constituents"})
        df = tables[0]
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        print(f"  Found {len(tickers)} tickers in S&P 500 constituent list (Wikipedia).")
        return tickers
    except Exception as exc:
        print(f"  Wikipedia fetch failed ({exc.__class__.__name__}: {exc}).")
        print(f"  Using built-in fallback list of {len(_FALLBACK_TICKERS)} major S&P 500 tickers.")
        return _FALLBACK_TICKERS


def _generate_synthetic_prices(tickers: list[str], n_days: int, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Generate realistic synthetic daily close prices with factor-driven correlation.

    Uses a two-factor model:
      - Market factor  (high positive correlation across all stocks)
      - Sector factor  (moderate correlation within sector groups)
    plus idiosyncratic noise, so the resulting PCA structure is meaningful.
    """
    rng = np.random.default_rng(seed)
    n = len(tickers)

    # Latent factor returns
    market = rng.normal(0.0003, 0.010, size=n_days)
    sector = rng.normal(0.0000, 0.008, size=(5, n_days))

    returns = np.empty((n_days, n))
    for i, _ in enumerate(tickers):
        sector_idx = i % 5
        beta_m = rng.uniform(0.6, 1.4)
        beta_s = rng.uniform(0.2, 0.8)
        idio = rng.normal(0.0000, 0.012, size=n_days)
        returns[:, i] = beta_m * market + beta_s * sector[sector_idx] + idio

    # Convert to price series (start at 100)
    prices = 100 * np.exp(np.cumsum(returns, axis=0))

    end = pd.Timestamp.today().normalize()
    dates = pd.bdate_range(end=end, periods=n_days)
    return pd.DataFrame(prices, index=dates, columns=tickers)


def download_prices(tickers: list[str], years: int = N_YEARS) -> pd.DataFrame:
    """Download adjusted-close prices for *tickers* over the last *years* years.

    Falls back to synthetic price data when network access is unavailable.
    """
    end = pd.Timestamp.today()
    start = end - pd.DateOffset(years=years)
    n_days = int(years * 252)  # approximate trading days
    print(f"  Downloading prices from {start.date()} to {end.date()} …")
    try:
        raw = yf.download(
            tickers,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        if raw.empty:
            raise ValueError("yfinance returned an empty DataFrame.")
        # yfinance returns a MultiIndex when >1 ticker is requested
        prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
        print(f"  Downloaded {prices.shape[1]} tickers × {prices.shape[0]} trading days.")
        return prices
    except Exception as exc:
        print(f"  yfinance download failed ({exc.__class__.__name__}: {exc}).")
        print(f"  Generating synthetic price data for {len(tickers)} tickers × {n_days} days.")
        return _generate_synthetic_prices(tickers, n_days)


def clean_returns(prices: pd.DataFrame, min_fraction: float = MIN_DATA_FRACTION) -> pd.DataFrame:
    """Compute log-returns and drop tickers / dates with insufficient data."""
    log_ret = np.log(prices / prices.shift(1)).iloc[1:]

    # Drop tickers with too many NaN values
    threshold = int(min_fraction * len(log_ret))
    log_ret = log_ret.dropna(axis=1, thresh=threshold)

    # Forward-fill remaining gaps (e.g. single missing days) then backfill edges
    log_ret = log_ret.ffill().bfill()

    # Drop any remaining columns that are still all-NaN
    log_ret = log_ret.dropna(axis=1, how="all")

    print(f"  Clean return matrix: {log_ret.shape[0]} days × {log_ret.shape[1]} stocks.")
    return log_ret


def run_pca(returns: pd.DataFrame, n_components: int = N_COMPONENTS):
    """Standardise *returns* and fit PCA; return (pca, scores_df, loadings_df)."""
    scaler = StandardScaler()
    X = scaler.fit_transform(returns)

    pca = PCA(n_components=min(n_components, X.shape[1]), random_state=RANDOM_STATE)
    scores = pca.fit_transform(X)

    pc_labels = [f"PC{i+1}" for i in range(pca.n_components_)]
    scores_df = pd.DataFrame(scores, index=returns.index, columns=pc_labels)
    loadings_df = pd.DataFrame(
        pca.components_.T,
        index=returns.columns,
        columns=pc_labels,
    )
    evr = pca.explained_variance_ratio_
    print(
        f"  PCA done. PC1 explains {evr[0]*100:.1f}% | "
        f"Top {n_components} PCs explain {evr.sum()*100:.1f}% of total variance."
    )
    return pca, scores_df, loadings_df


# ── Plotting functions ─────────────────────────────────────────────────────────

def plot_scree(pca: PCA, out_dir: str) -> None:
    """Bar chart of explained-variance ratio per component."""
    evr = pca.explained_variance_ratio_ * 100
    pcs = [f"PC{i+1}" for i in range(len(evr))]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(pcs, evr, color=sns.color_palette("Blues_d", len(evr)))
    ax.set_xlabel("Principal Component", fontsize=13)
    ax.set_ylabel("Explained Variance (%)", fontsize=13)
    ax.set_title("Scree Plot – S&P 500 PCA", fontsize=15, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))

    for bar, val in zip(bars, evr):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    path = os.path.join(out_dir, "scree_plot.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_cumulative_variance(pca: PCA, out_dir: str) -> None:
    """Line chart of cumulative explained variance."""
    cum_evr = np.cumsum(pca.explained_variance_ratio_) * 100
    pcs = range(1, len(cum_evr) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(list(pcs), cum_evr, marker="o", linewidth=2, color="#2196F3")
    ax.fill_between(list(pcs), cum_evr, alpha=0.15, color="#2196F3")

    for threshold in (50, 75, 90):
        if cum_evr[-1] >= threshold:
            ax.axhline(threshold, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.text(len(pcs) + 0.1, threshold + 0.5, f"{threshold}%", color="grey", fontsize=9)

    ax.set_xlabel("Number of Principal Components", fontsize=13)
    ax.set_ylabel("Cumulative Explained Variance (%)", fontsize=13)
    ax.set_title("Cumulative Explained Variance – S&P 500 PCA", fontsize=15, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.set_xticks(list(pcs))

    plt.tight_layout()
    path = os.path.join(out_dir, "cumulative_variance.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_top_loadings(loadings_df: pd.DataFrame, out_dir: str, n_pcs: int = 4, top_n: int = TOP_N_LOADINGS) -> None:
    """Horizontal bar chart of the top *top_n* stock loadings for the first *n_pcs* PCs."""
    n_pcs = min(n_pcs, len(loadings_df.columns))
    fig, axes = plt.subplots(1, n_pcs, figsize=(5 * n_pcs, 8), sharey=False)
    if n_pcs == 1:
        axes = [axes]

    for ax, pc in zip(axes, loadings_df.columns[:n_pcs]):
        col = loadings_df[pc].abs().nlargest(top_n).index
        vals = loadings_df.loc[col, pc].sort_values()
        colors = ["#E53935" if v < 0 else "#1E88E5" for v in vals]
        ax.barh(vals.index, vals.values, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(pc, fontsize=12, fontweight="bold")
        ax.set_xlabel("Loading", fontsize=10)
        ax.tick_params(axis="y", labelsize=8)

    fig.suptitle(f"Top {top_n} Stock Loadings per Principal Component", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "top_loadings.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_biplot(scores_df: pd.DataFrame, loadings_df: pd.DataFrame, pca: PCA, out_dir: str, n_arrows: int = N_BIPLOT_ARROWS) -> None:
    """Scatter of PC1 vs PC2 observation scores with the top loading vectors overlaid."""
    pc1_scores = scores_df["PC1"].values
    pc2_scores = scores_df["PC2"].values

    # Scale scores to [-1, 1] for visual clarity
    scale1 = 1.0 / (pc1_scores.max() - pc1_scores.min())
    scale2 = 1.0 / (pc2_scores.max() - pc2_scores.min())

    # Choose the stocks with the largest combined loading magnitude
    combined = (loadings_df["PC1"] ** 2 + loadings_df["PC2"] ** 2) ** 0.5
    top_stocks = combined.nlargest(n_arrows).index

    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(
        pc1_scores * scale1,
        pc2_scores * scale2,
        c=range(len(pc1_scores)),
        cmap="coolwarm",
        alpha=0.4,
        s=8,
    )
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Trading Day Index", fontsize=9)

    for ticker in top_stocks:
        lx = loadings_df.loc[ticker, "PC1"]
        ly = loadings_df.loc[ticker, "PC2"]
        ax.annotate(
            "",
            xy=(lx, ly),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color="darkred", lw=1.2),
        )
        ax.text(lx * 1.05, ly * 1.05, ticker, fontsize=6.5, color="darkred", ha="center")

    evr = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1  ({evr[0]*100:.1f}% variance explained)", fontsize=12)
    ax.set_ylabel(f"PC2  ({evr[1]*100:.1f}% variance explained)", fontsize=12)
    ax.set_title("PCA Biplot – S&P 500 Daily Returns\n(dots = trading days, arrows = stock loadings)", fontsize=13, fontweight="bold")
    ax.axhline(0, color="grey", linewidth=0.6, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.6, linestyle="--")

    plt.tight_layout()
    path = os.path.join(out_dir, "biplot.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("\n=== S&P 500 PCA ===\n")

    print("Step 1/4 – Fetching S&P 500 tickers …")
    tickers = fetch_sp500_tickers()

    print("\nStep 2/4 – Downloading price history …")
    prices = download_prices(tickers)

    print("\nStep 3/4 – Computing returns and running PCA …")
    returns = clean_returns(prices)
    pca, scores_df, loadings_df = run_pca(returns, n_components=N_COMPONENTS)

    print("\nStep 4/4 – Generating charts …")
    plot_scree(pca, OUTPUT_DIR)
    plot_cumulative_variance(pca, OUTPUT_DIR)
    plot_top_loadings(loadings_df, OUTPUT_DIR)
    plot_biplot(scores_df, loadings_df, pca, OUTPUT_DIR)

    # ── Summary table ──────────────────────────────────────────────────────────
    evr = pca.explained_variance_ratio_
    cum_evr = np.cumsum(evr)
    summary = pd.DataFrame(
        {
            "Explained Variance (%)": evr * 100,
            "Cumulative Variance (%)": cum_evr * 100,
        },
        index=[f"PC{i+1}" for i in range(len(evr))],
    )
    summary_path = os.path.join(OUTPUT_DIR, "pca_summary.csv")
    summary.to_csv(summary_path, float_format="%.4f")
    print(f"  Saved: {summary_path}")

    print("\n── PCA Summary ──")
    print(summary.to_string(float_format=lambda x: f"{x:.2f}%"))

    print(f"\n✓ All outputs written to ./{OUTPUT_DIR}/\n")


if __name__ == "__main__":
    main()
