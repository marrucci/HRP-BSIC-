import pickle
import pandas as pd

# Load pickle file
with open("/Users/tommasomarrucci/Desktop/Article HRP/monster_ohlcv.pkl", "rb") as file:
    data = pickle.load(file)

# Extract "Adj Close" column for tickers that have it
full_data_tickers = {
    ticker: df["Adj Close"] for ticker, df in data.items() if "Adj Close" in df.columns
}

# Convert to DataFrame
adj_close_df = pd.DataFrame(full_data_tickers)

# Drop columns (tickers) that contain any NaN values
adj_close_df = adj_close_df.dropna(axis=1)

# Save to CSV
adj_close_df.to_csv("adj_close_data_clean.csv", index=True)  # Keeping index to maintain time order

# Save to CSV in the same folder
#output_path = "/Users/tommasomarrucci/Desktop/Article HRP/adj_close_data_clean.csv"
#adj_close_df.to_csv(output_path, index=True)  # Keeping index to maintain time order

#rint(f"Saved adjusted close prices for {len(adj_close_df.columns)} tickers at {output_path} after dropping columns with NaN values.")