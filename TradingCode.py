"""machine_learning_trading_project.py

Simple ML project for learning trading strategies based on technical indicators.
Author: ChatGPT o3
Date: 2025‑06‑23
Dependencies:
    pip install yfinance pandas scikit-learn matplotlib xgboost
    # Optional: pandas_ta (older versions) if you prefer using the library

This script:
1. Downloads historical OHLCV data using yfinance.
2. Computes RSI, SMA, MACD indicators with built-in helper functions.
3. Generates BUY / SELL / HOLD labels from a straightforward rule‑based strategy.
4. Trains a RandomForestClassifier (or XGBoost) to imitate the strategy (behavior cloning).
5. Evaluates the model on unseen data and compares returns with the original strategy using a vectorized back‑tester.
6. Includes a very simple brute‑force optimiser for the strategy’s thresholds.

Everything is written for clarity and educational purposes – not production HFT!

Run:
    python machine_learning_trading_project.py
"""

import warnings
warnings.filterwarnings("ignore")

# ========= 1. Imports ========= #
import yfinance as yf
import pandas as pd
# pandas_ta is optional and currently incompatible with numpy>=2.
# We implement the few indicators we need locally to avoid this dependency.
# Core ML dependencies
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
# You can switch to XGBoost by uncommenting the line below (needs xgboost installed)
# from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# ========= 2. Helper functions ========= #
def fetch_data(ticker: str = "AAPL", period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """Download historical data from Yahoo Finance."""
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError("No data downloaded – check your ticker or internet connection.")
    return df


def sma(series: pd.Series, length: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window=length, min_periods=length).mean()


def ema(series: pd.Series, length: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=length, adjust=False).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(span=length, adjust=False).mean()
    roll_down = down.ewm(span=length, adjust=False).mean()
    rs = roll_up / roll_down
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Moving Average Convergence Divergence."""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    # Explicitly pass the original series index to avoid issues when pandas
    # interprets the values as scalars. This ensures the MACD components are
    # properly aligned with the input data across different pandas versions.
    return pd.DataFrame({
        "MACD": macd_line,
        "MACD_signal": signal_line,
        "MACD_hist": hist
    }, index=series.index)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add RSI(14), SMA(50), SMA(200) and MACD columns to the DataFrame."""
    df = df.copy()

    df["RSI"] = rsi(df["Close"], length=14)
    df["SMA50"] = sma(df["Close"], length=50)
    df["SMA200"] = sma(df["Close"], length=200)

    macd_df = macd(df["Close"], fast=12, slow=26, signal=9)
    df[["MACD", "MACD_signal", "MACD_hist"]] = macd_df

    return df.dropna()


def rule_based_signals(df: pd.DataFrame,
                       rsi_oversold: int = 30,
                       rsi_overbought: int = 70,
                       sma_fast: str = "SMA50",
                       sma_slow: str = "SMA200",
                       macd_col: str = "MACD_hist") -> pd.DataFrame:
    """Generate Buy(+1), Sell(-1) & Hold(0) labels.\n
    Buy: RSI < oversold AND price above fast SMA AND MACD_hist turns positive\n
    Sell: RSI > overbought OR price crosses below slow SMA OR MACD_hist turns negative\n
    Hold otherwise."""
    df = df.copy()
    df["Action"] = 0  # Default Hold

    buy_cond = (df["RSI"] < rsi_oversold) & (df["Close"] > df[sma_fast]) & (df[macd_col] > 0)
    sell_cond = (df["RSI"] > rsi_overbought) | (df["Close"] < df[sma_slow]) | (df[macd_col] < 0)

    df.loc[buy_cond, "Action"] = 1
    df.loc[sell_cond, "Action"] = -1

    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return ML-ready feature matrix X and target y."""
    feature_cols = ["RSI", "SMA50", "SMA200", "MACD", "MACD_signal", "MACD_hist"]
    X = df[feature_cols]
    y = df["Action"]
    return X, y


def train_model(X_train: pd.DataFrame, y_train: pd.Series, model_type: str = "rf"):
    """Train and return a RandomForest (default) or XGBoost model."""
    if model_type == "xgb":
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss"
        )
    else:
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1
        )
    model.fit(X_train, y_train)
    return model


def make_predictions(model, X: pd.DataFrame) -> np.ndarray:
    """Return predicted actions."""
    return model.predict(X)


# ========= 3. Backtesting ========= #
def backtest(df: pd.DataFrame, actions_col: str = "Action", initial_cash: float = 10_000.0) -> pd.Series:
    """Very simple vectorised back‑tester assuming fully invested/not invested and no shorting.\n
    • action == 1 (Buy)  => go long all cash\n
    • action == -1 (Sell) => liquidate to cash\n
    • Hold keeps existing position.\n
    Returns a series of portfolio values."""
    cash = initial_cash
    shares = 0.0
    portfolio_values = []

    for price, action in zip(df["Close"], df[actions_col]):
        # Buy signal – invest all cash
        if action == 1 and cash > 0:
            shares = cash / price
            cash = 0
        # Sell signal – liquidate
        elif action == -1 and shares > 0:
            cash = shares * price
            shares = 0
        # Portfolio value
        portfolio_values.append(cash + shares * price)

    return pd.Series(portfolio_values, index=df.index, name="Portfolio")


def plot_results(df: pd.DataFrame, base_col: str, model_col: str, ticker: str):
    """Plot cumulative returns of baseline and model portfolios."""
    plt.figure(figsize=(12, 6))
    df[base_col].pct_change().add(1).cumprod().plot(label="Rule‑based Strategy")
    df[model_col].pct_change().add(1).cumprod().plot(label="ML Model")

    plt.title(f"Cumulative Returns – {ticker}")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ========= 4. Simple optimiser ========= #
def optimise_thresholds(df: pd.DataFrame,
                        rsi_range=range(20, 40, 2),
                        overbought_range=range(60, 80, 2)) -> tuple[int, int, float]:
    """Brute force search for RSI oversold/overbought levels that maximise strategy CAGR."""
    best_cagr = -np.inf
    best_pair = (30, 70)

    years = (df.index[-1] - df.index[0]).days / 365.25

    for low in rsi_range:
        for high in overbought_range:
            if low >= high:
                continue
            tmp = rule_based_signals(df, rsi_oversold=low, rsi_overbought=high)
            portfolio = backtest(tmp)
            cagr = (portfolio.iloc[-1] / portfolio.iloc[0]) ** (1 / years) - 1
            if cagr > best_cagr:
                best_cagr = cagr
                best_pair = (low, high)

    return (*best_pair, best_cagr)


# ========= 5. Main flow ========= #
def main():
    ticker = "AAPL"
    period = "10y"  # Feel free to change
    df = fetch_data(ticker, period=period)
    df = add_indicators(df)

    # 5.1 Optimise strategy thresholds (optional)
    best_low, best_high, best_cagr = optimise_thresholds(df)
    print(f"Best RSI thresholds found: oversold={best_low}, overbought={best_high} (CAGR={best_cagr:.2%})")

    df = rule_based_signals(df, rsi_oversold=best_low, rsi_overbought=best_high)

    ### Prepare ML data ###
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        shuffle=False)  # time series split

    model = train_model(X_train, y_train, model_type="rf")

    ### Evaluate ###
    y_pred = make_predictions(model, X_test)
    print("Model vs. Rule‑based – test period")
    print(classification_report(y_test, y_pred, target_names=["Sell", "Hold", "Buy"]))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    ### Backtest both strategies ###
    df.loc[X_test.index, "ML_Action"] = y_pred  # align predictions with df

    baseline_portfolio = backtest(df, "Action")
    model_portfolio = backtest(df, "ML_Action")

    df["Baseline_Portfolio"] = baseline_portfolio
    df["Model_Portfolio"] = model_portfolio

    # Plot results
    plot_results(df, "Baseline_Portfolio", "Model_Portfolio", ticker)

    # Print final returns
    base_return = baseline_portfolio.iloc[-1] / baseline_portfolio.iloc[0] - 1
    model_return = model_portfolio.iloc[-1] / model_portfolio.iloc[0] - 1
    print(f"Baseline total return: {base_return:.2%}")
    print(f"Model total return:    {model_return:.2%}")

if __name__ == "__main__":
    main()
