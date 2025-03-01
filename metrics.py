import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import mpl_bsic as bsic
from mpl_bsic import apply_bsic_style, apply_bsic_logo

# ------------------------------
# 1. Download IRX (13-week T-Bill yield) for Risk-Free Rate
# ------------------------------
irx_df = yf.download("^IRX", period="1d")
if not irx_df.empty:
    # IRX is quoted as a percentage (e.g., 4.50), so convert to decimal.
    risk_free_rate = float(irx_df["Close"].iloc[-1]) / 100
else:
    risk_free_rate = 0.00
print(f"Current IRX risk-free rate: {risk_free_rate:.4f}")

# ------------------------------
# 2. Load HRP Portfolio Data
# ------------------------------
hrp_df = pd.read_csv("hrp_portfolio_returns.csv", index_col="Date", parse_dates=True)
if "Daily_Return" not in hrp_df.columns:
    hrp_df["Daily_Return"] = hrp_df["Cumulative Return"].pct_change().fillna(0)

# ------------------------------
# 3. Load Market Data (SP500)
# ------------------------------
sp500_df = pd.read_csv("sp500_cumulative_returns.csv", index_col="Date", parse_dates=True)
if "Daily_Return" not in sp500_df.columns:
    sp500_df["Daily_Return"] = sp500_df["Cumulative Return"].pct_change().fillna(0)

# Align dates between HRP and SP500 data
combined_df = hrp_df.join(
    sp500_df[["Daily_Return"]].rename(columns={"Daily_Return": "Market_Daily_Return"}),
    how="inner"
)

# ------------------------------
# 4. Set Parameters
# ------------------------------
trading_days = 252          # Number of trading days in a year
rolling_window = 60         # 90-day rolling window

# ------------------------------
# 5. Calculate Risk Metrics & Rolling Sharpe Ratio
# ------------------------------
daily_rf = risk_free_rate / trading_days
combined_df["Excess_Return"] = combined_df["Daily_Return"] - daily_rf

# Rolling Sharpe (annualized)
combined_df["Rolling_Mean"] = combined_df["Excess_Return"].rolling(window=rolling_window).mean()
combined_df["Rolling_Std"] = combined_df["Excess_Return"].rolling(window=rolling_window).std()
combined_df["Rolling_Sharpe"] = (combined_df["Rolling_Mean"] / combined_df["Rolling_Std"]) * np.sqrt(trading_days)

# Total Percentage Return
initial_value = combined_df["Cumulative Return"].iloc[0]
final_value = combined_df["Cumulative Return"].iloc[-1]
total_return_pct = ((final_value / initial_value) - 1) * 100

# Full-Period Mean Sharpe Ratio
full_period_sharpe = (combined_df["Excess_Return"].mean() / combined_df["Excess_Return"].std()) * np.sqrt(trading_days)

# Additional Risk Metrics
covariance = np.cov(combined_df["Daily_Return"], combined_df["Market_Daily_Return"])[0, 1]
market_variance = np.var(combined_df["Market_Daily_Return"])
beta = covariance / market_variance if market_variance != 0 else np.nan
avg_portfolio_return = combined_df["Daily_Return"].mean() * trading_days
treynor_ratio = (avg_portfolio_return - risk_free_rate) / beta if beta else np.nan

downside_returns = combined_df["Excess_Return"][combined_df["Excess_Return"] < 0]
downside_std = downside_returns.std()
avg_excess_return = avg_portfolio_return - risk_free_rate
sortino_ratio = (avg_excess_return / (downside_std * np.sqrt(trading_days))
                 if downside_std != 0 else np.nan)

portfolio_series = combined_df["Cumulative Return"]
rolling_max = portfolio_series.cummax()
drawdown = (portfolio_series - rolling_max) / rolling_max
max_drawdown = drawdown.min()

std_rolling_sharpe = combined_df["Rolling_Sharpe"].std()

# ------------------------------
# 6. Print Risk Metrics
# ------------------------------
print("\nRisk Metrics:")
print("------------------------------")
print(f"Treynor Ratio: {treynor_ratio:.4f}")
print(f"Sortino Ratio: {sortino_ratio:.4f}")
print(f"Maximum Drawdown: {max_drawdown:.4%}")
print(f"Std Dev of Rolling Sharpe Ratio: {std_rolling_sharpe:.4f}")
print(f"Mean Sharpe Ratio (Full Period): {full_period_sharpe:.4f}")
print(f"Total Return Percentage: {total_return_pct:.2f}%")

# ------------------------------
# 7. Plot the 60-Day Rolling Sharpe Ratio with BSIC Styling
# ------------------------------
fig, ax = plt.subplots(figsize=(10, 5), dpi=500)
apply_bsic_style(fig, ax)

ax.plot(combined_df.index, combined_df["Rolling_Sharpe"], label="60-Day Rolling Sharpe Ratio")
ax.set_xlabel("Date")
ax.set_ylabel("Sharpe Ratio")
ax.set_title("60-Day Rolling Sharpe Ratio")
ax.legend()
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
fig.tight_layout()

# Apply BSIC logo (if it works; otherwise you may comment this out)
apply_bsic_logo(fig, ax)

# Optionally, export the figure
bsic.export_figure(fig, "rolling_sharpe_ratio_plot.png")
plt.show()
