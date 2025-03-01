import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ============================
# 1. Load HRP Portfolio Data
# ============================
# This CSV is created by your HRP backtest code, with columns:
#   "Date" (index) and "Cumulative Return"
hrp_df = pd.read_csv("hrp_portfolio_returns.csv", index_col="Date", parse_dates=True)

# Compute daily returns if not already in the file
if "Daily_Return" not in hrp_df.columns:
    hrp_df["Daily_Return"] = hrp_df["Cumulative Return"].pct_change().fillna(0)

# ============================
# 2. Load Market (e.g. SP500) Data
# ============================
# Must have a similar structure with "Date" as index and "Cumulative Return"
sp500_df = pd.read_csv("sp500_cumulative_returns.csv", index_col="Date", parse_dates=True)

# Compute daily returns if not already in the file
if "Daily_Return" not in sp500_df.columns:
    sp500_df["Daily_Return"] = sp500_df["Cumulative Return"].pct_change().fillna(0)

# Align the two DataFrames on dates
combined_df = hrp_df.join(
    sp500_df[["Daily_Return"]].rename(columns={"Daily_Return": "Market_Daily_Return"}),
    how="inner"
)

# ============================
# 3. Set Parameters
# ============================
risk_free_rate = 0.00  # annual risk-free rate (in decimal)
trading_days = 252
rolling_window = 252

# ============================
# 4. Rolling Sharpe Computation
# ============================
daily_rf = risk_free_rate / trading_days
combined_df["Excess_Return"] = combined_df["Daily_Return"] - daily_rf

# Rolling mean and std of daily excess returns
combined_df["Rolling_Mean"] = combined_df["Excess_Return"].rolling(window=rolling_window).mean()
combined_df["Rolling_Std"] = combined_df["Excess_Return"].rolling(window=rolling_window).std()
combined_df["Rolling_Sharpe"] = (
    combined_df["Rolling_Mean"] / combined_df["Rolling_Std"]
) * np.sqrt(trading_days)

# ============================
# 5. Compute Other Risk Metrics
# ============================

# ---- (A) Treynor Ratio (non-rolling) ----
# Treynor = (Annualized portfolio return - Risk-free rate) / Beta
covariance = np.cov(combined_df["Daily_Return"], combined_df["Market_Daily_Return"])[0, 1]
market_variance = np.var(combined_df["Market_Daily_Return"])
beta = covariance / market_variance if market_variance != 0 else np.nan
avg_portfolio_return = combined_df["Daily_Return"].mean() * trading_days
treynor_ratio = (avg_portfolio_return - risk_free_rate) / beta if beta else np.nan

# ---- (B) Sortino Ratio (non-rolling) ----
downside_returns = combined_df["Excess_Return"][combined_df["Excess_Return"] < 0]
downside_std = downside_returns.std()
avg_excess_return = avg_portfolio_return - risk_free_rate
sortino_ratio = (
    avg_excess_return / (downside_std * np.sqrt(trading_days))
    if downside_std != 0
    else np.nan
)

# ---- (C) Maximum Drawdown ----
portfolio_series = combined_df["Cumulative Return"]
rolling_max = portfolio_series.cummax()
drawdown = (portfolio_series - rolling_max) / rolling_max
max_drawdown = drawdown.min()

# ---- (D) Std Dev of Rolling Sharpe Ratio ----
std_rolling_sharpe = combined_df["Rolling_Sharpe"].std()

# ---- (E) Mean Sharpe Ratio Over Full Period ----
#   i.e. (mean of daily excess returns / std of daily excess returns) * sqrt(252)
full_period_sharpe = (
    combined_df["Excess_Return"].mean() / combined_df["Excess_Return"].std()
) * np.sqrt(trading_days)

# ============================
# 6. Print Risk Metrics
# ============================
print("\nRisk Metrics:")
print("------------------------------")
print(f"Treynor Ratio: {treynor_ratio:.4f}")
print(f"Sortino Ratio: {sortino_ratio:.4f}")
print(f"Maximum Drawdown: {max_drawdown:.4%}")
print(f"Std Dev of Rolling Sharpe Ratio: {std_rolling_sharpe:.4f}")
print(f"Mean Sharpe Ratio (Full Period): {full_period_sharpe:.4f}")

# ============================
# 7. Plot Rolling Sharpe Ratio
# ============================
plt.figure(figsize=(10, 5))
plt.plot(combined_df.index, combined_df["Rolling_Sharpe"], label="Rolling Sharpe Ratio")
plt.xlabel("Date")
plt.ylabel("Sharpe Ratio")
plt.title("Rolling Sharpe Ratio")
plt.legend()
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()