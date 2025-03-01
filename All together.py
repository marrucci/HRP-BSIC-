#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 03:00:00 2025

Combined plot for SP500, Non-Rebalanced, Rebalanced, and HRP portfolios.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import mpl_bsic as bsic  # Import the BSIC styling library
from mpl_bsic import apply_bsic_style, apply_bsic_logo


# ------------------------------
# Load CSV Data
# ------------------------------
# SP500 cumulative returns
sp500 = pd.read_csv("sp500_cumulative_returns.csv", index_col=0, parse_dates=True)
# Non-Rebalanced Portfolio cumulative returns
non_rebalanced = pd.read_csv("non_rebalanced_cumulative_returns.csv", index_col=0, parse_dates=True)
# Rebalanced Portfolio cumulative returns
rebalanced = pd.read_csv("rebalanced_cumulative_returns.csv", index_col=0, parse_dates=True)
# HRP Portfolio cumulative returns
hrp = pd.read_csv("hrp_portfolio_returns.csv", index_col=0, parse_dates=True)

# ------------------------------
# Plotting
# ------------------------------
fig, ax1 = plt.subplots(figsize=(12, 6), dpi=500)

# Apply BSIC styling
bsic.apply_bsic_style(fig, ax1)

# Plot all series
ax1.plot(non_rebalanced.index, non_rebalanced["Cumulative Return"],
         label="Non-Rebalanced Portfolio")
ax1.plot(rebalanced.index, rebalanced["Cumulative Return"],
         label="Rebalanced Portfolio")
ax1.plot(sp500.index, sp500["Cumulative Return"],
         label="SP500")
ax1.plot(hrp.index, hrp["Cumulative Return"],
         label="HRP Portfolio")

# Set axis labels and title
ax1.set_xlabel("Date")
ax1.set_ylabel("Cumulative Return")
plt.title("Portfolio Cumulative Returns Comparison")

# X-axis: major ticks every year, with rotated labels for clarity
ax1.xaxis.set_major_locator(mdates.YearLocator(1))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

# Y-axis: ticks every 1 unit
ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))

# Apply BSIC logo styling
bsic.apply_bsic_logo(fig, ax1)

# Move legend to bottom right
plt.legend(loc='lower right')

fig.tight_layout()
bsic.export_figure(fig, "combined_portfolio_returns_plot.png")
plt.show()