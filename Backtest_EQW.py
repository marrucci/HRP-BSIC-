import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import mpl_bsic as bsic  # Importing the BSIC styling library
import yfinance as yf

# Load the cleaned adjusted close prices
adj_close_df = pd.read_csv("adj_close_data_clean.csv", index_col=0, parse_dates=True)

# Compute daily returns
returns_df = adj_close_df.pct_change().dropna()

# Number of stocks in the portfolio
num_stocks = returns_df.shape[1]

# ------------------------------
# SP500
# ------------------------------
# Download S&P 500 data
sp500 = yf.download("^GSPC", start="2005-02-11", end="2025-02-26")["Close"]

# Compute daily returns
sp500_returns = sp500.pct_change().dropna()
# Compute cumulative returns
sp500_cumulative_returns = (1 + sp500_returns).cumprod()

sp500_cumulative_returns.to_csv("sp500_cumulative_returns.csv", header=["Cumulative Return"])
# ------------------------------
# Non-Rebalanced Portfolio
# ------------------------------

weights = 1 / num_stocks  
equal_weighted_portfolio_return = returns_df.mul(weights).sum(axis=1)
cumulative_returns_non_rebalanced = (1 + equal_weighted_portfolio_return).cumprod()

# ------------------------------
# Rebalanced Portfolio (Rebalance every first tradable day of December)
# ------------------------------
december_data = returns_df[returns_df.index.month == 12]
rebalancing_days = december_data.groupby(december_data.index.year).head(1).index

portfolio_value = 1.0
rebalanced_values = []
dates = []

holdings = np.full(num_stocks, portfolio_value / num_stocks)

for date, daily_returns in returns_df.iterrows():
    if date in rebalancing_days:
        holdings = np.full(num_stocks, portfolio_value / num_stocks)
    
    holdings *= (1 + daily_returns.values)
    portfolio_value = holdings.sum()
    
    rebalanced_values.append(portfolio_value)
    dates.append(date)

cumulative_returns_rebalanced = pd.Series(rebalanced_values, index=dates)

# ------------------------------
# Plotting
# ------------------------------
fig, ax1 = plt.subplots(figsize=(12, 6), dpi=500)

bsic.apply_bsic_style(fig, ax1)

# Plot both portfolios
ax1.plot(cumulative_returns_non_rebalanced.index, cumulative_returns_non_rebalanced,
         label="Non-Rebalanced Portfolio")
ax1.plot(cumulative_returns_rebalanced.index, cumulative_returns_rebalanced,
         label="Rebalanced Portfolio")
ax1.plot(sp500_cumulative_returns.index, sp500_cumulative_returns,
         label="SP500")

ax1.set_xlabel("Date")
ax1.set_ylabel("Cumulative Return")

# X-axis: major ticks every year, minor ticks every month
ax1.xaxis.set_major_locator(mdates.YearLocator(1))
#ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
#ax1.xaxis.set_minor_locator(mdates.MonthLocator())
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

# Y-axis: ticks every 0.5, with minor ticks in between
ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))
#ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())

bsic.apply_bsic_logo(fig, ax1)

plt.title("Portfolio Cumulative Returns: Rebalanced vs Non-Rebalanced")

# Move legend to bottom right
plt.legend(loc='lower right')

fig.tight_layout()
bsic.export_figure(fig, "portfolio_returns_plot.png")
plt.show()