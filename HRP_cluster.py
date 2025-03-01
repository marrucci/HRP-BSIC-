import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.covariance import LedoitWolf
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import mpl_bsic as bsic  # Import the BSIC styling library
from mpl_bsic import apply_bsic_style, apply_bsic_logo

# For reproducibility of random sampling
np.random.seed(42)

# ===============================
# 1. Load Data and Compute Returns
# ===============================
adj_close_df = pd.read_csv("adj_close_data_clean.csv", index_col=0, parse_dates=True)
returns_df = adj_close_df.pct_change().dropna()

# ===============================
# 2. Define Annual Rebalancing Days (first trading day of December)
# ===============================
december_data = returns_df[returns_df.index.month == 12]
rebalancing_days = december_data.groupby(december_data.index.year).head(1).index.sort_values()

all_dates = returns_df.index.sort_values()
final_date = all_dates[-1]
rebalance_schedule = list(rebalancing_days) + [final_date]

# ===============================
# 3. Helper Functions for HRP and Clustering
# ===============================
def get_shrunk_covariance(past_returns):
    lw = LedoitWolf()
    lw.fit(past_returns)
    cov = lw.covariance_
    return pd.DataFrame(cov, index=past_returns.columns, columns=past_returns.columns)

def get_quasi_diag(link_matrix, n_assets):
    link_matrix = link_matrix.astype(int)
    sort_ix = list(link_matrix[-1, :2])
    while any(i >= n_assets for i in sort_ix):
        sort_ix_new = []
        for i in sort_ix:
            if i >= n_assets:
                idx = i - n_assets
                sort_ix_new.extend(link_matrix[idx, :2].tolist())
            else:
                sort_ix_new.append(i)
        sort_ix = sort_ix_new
    return sort_ix

def get_cluster_variance(cov, cluster):
    sub_cov = cov.loc[cluster, cluster]
    diag_vals = np.maximum(np.diag(sub_cov), 0)
    inv_diag = 1.0 / diag_vals
    weights = inv_diag / inv_diag.sum()
    return np.dot(weights, np.dot(sub_cov, weights))

def compute_hrp_weights(cov):
    # If there's only one asset, return a weight of 1.
    if cov.shape[0] == 1:
        return pd.Series(1.0, index=cov.index)
    diag_vals = np.maximum(np.diag(cov), 0)
    std_devs = np.sqrt(diag_vals)
    corr = cov / np.outer(std_devs, std_devs)
    dist = np.sqrt((1 - corr) / 2)
    dist_condensed = squareform(dist.values, checks=False)
    link_matrix = linkage(dist_condensed, method='ward')
    
    sorted_idx = get_quasi_diag(link_matrix, len(cov))
    sorted_assets = cov.index[sorted_idx]
    
    weights = pd.Series(1.0, index=sorted_assets)
    clusters = [list(sorted_assets)]
    while any(len(cluster) > 1 for cluster in clusters):
        new_clusters = []
        for cluster in clusters:
            if len(cluster) == 1:
                new_clusters.append(cluster)
                continue
            split = len(cluster) // 2
            c1, c2 = cluster[:split], cluster[split:]
            var1 = get_cluster_variance(cov, c1)
            var2 = get_cluster_variance(cov, c2)
            alpha = 1 - var1 / (var1 + var2)
            weights.loc[c1] *= alpha
            weights.loc[c2] *= (1 - alpha)
            new_clusters.extend([c1, c2])
        clusters = new_clusters
    return weights.reindex(cov.index, fill_value=0)

def get_clusters(cov, num_clusters=8):
    diag_vals = np.maximum(np.diag(cov), 0)
    std_devs = np.sqrt(diag_vals)
    corr = cov / np.outer(std_devs, std_devs)
    dist = np.sqrt((1 - corr) / 2)
    dist_condensed = squareform(dist.values, checks=False)
    link_matrix = linkage(dist_condensed, method='average')
    
    clusters = fcluster(link_matrix, num_clusters, criterion='maxclust')
    cluster_assignments = pd.DataFrame({'Stock': cov.index, 'Cluster': clusters})
    return cluster_assignments

def compute_hrp_weights_two_clusters(cov):
    """
    Computes HRP weights by splitting the asset universe into exactly 2 clusters.
    If one cluster ends up empty, the function falls back to standard HRP.
    """
    clusters_df = get_clusters(cov, num_clusters=2)
    cluster1 = clusters_df[clusters_df['Cluster'] == 1]['Stock'].tolist()
    cluster2 = clusters_df[clusters_df['Cluster'] == 2]['Stock'].tolist()
    
    if len(cluster1) == 0 or len(cluster2) == 0:
        print("Warning: one cluster is empty, reverting to standard HRP.")
        return compute_hrp_weights(cov)
    
    cov_cluster1 = cov.loc[cluster1, cluster1]
    cov_cluster2 = cov.loc[cluster2, cluster2]
    
    if cov_cluster1.empty or cov_cluster2.empty:
        print("Warning: one covariance matrix is empty, reverting to standard HRP.")
        return compute_hrp_weights(cov)
    
    weights_cluster1 = compute_hrp_weights(cov_cluster1)
    weights_cluster2 = compute_hrp_weights(cov_cluster2)
    
    weights_cluster1 = weights_cluster1 / weights_cluster1.sum()
    weights_cluster2 = weights_cluster2 / weights_cluster2.sum()
    
    combined = pd.Series(0, index=cov.index)
    combined.loc[cluster1] = 0.5 * weights_cluster1
    combined.loc[cluster2] = 0.5 * weights_cluster2
    
    return combined

# ===============================
# 4. Annual Rebalancing, Clustering & Backtest with 2 Clusters
# ===============================
portfolio_value = 1.0
dates_all, values_all = [], []
clusters_history = {}

last_shrunk_cov = None
last_weights = None
last_link_matrix = None

force_two_clusters = True

for i in range(len(rebalance_schedule) - 1):
    rebalance_day = rebalance_schedule[i]
    next_period_end = rebalance_schedule[i+1]
    
    lookback_start = rebalance_day - pd.DateOffset(years=1)
    if lookback_start < returns_df.index[0]:
        lookback_start = returns_df.index[0]
    
    past_returns = returns_df.loc[lookback_start:rebalance_day].dropna()
    if past_returns.shape[0] < 10:
        print(f"Skipping {rebalance_day} due to insufficient data.")
        continue
    
    shrunk_cov = get_shrunk_covariance(past_returns)
    last_shrunk_cov = shrunk_cov.copy()
    
    # For informational purposes, get 2-cluster assignments.
    clusters_df = get_clusters(shrunk_cov, num_clusters=2)
    clusters_history[rebalance_day] = clusters_df
    print(f"2-Cluster assignments on {rebalance_day.date()}:")
    print(clusters_df.sort_values('Cluster').to_string(index=False))
    
    if force_two_clusters:
        weights = compute_hrp_weights_two_clusters(shrunk_cov)
    else:
        weights = compute_hrp_weights(shrunk_cov)
    last_weights = weights.copy()
    
    diag_vals = np.maximum(np.diag(shrunk_cov), 0)
    std_devs = np.sqrt(diag_vals)
    corr = shrunk_cov / np.outer(std_devs, std_devs)
    dist = np.sqrt((1 - corr) / 2)
    dist_condensed = squareform(dist.values, checks=False)
    last_link_matrix = linkage(dist_condensed, method='ward')
    
    sub_period = returns_df.loc[rebalance_day:next_period_end].index
    if len(sub_period) <= 1:
        continue
    for day_idx in range(len(sub_period) - 1):
        current_day = sub_period[day_idx]
        next_day = sub_period[day_idx + 1]
        day_return = returns_df.loc[current_day]
        portfolio_ret = (day_return * weights).sum()
        portfolio_value *= (1 + portfolio_ret)
        dates_all.append(next_day)
        values_all.append(portfolio_value)

# ===============================
# 5. Display Last Covariance Matrix, Quasi-Diagonalized Matrix, Dendrogram & Weights for 50 Random Stocks (without ticker labels)
# ===============================
if last_shrunk_cov is not None:
    stocks = last_shrunk_cov.index.tolist()
    if len(stocks) > 50:
        sample_indices = np.random.choice(range(len(stocks)), size=50, replace=False)
    else:
        sample_indices = list(range(len(stocks)))
    
    sample_stocks = [stocks[i] for i in sample_indices]
    cov_sample = last_shrunk_cov.loc[sample_stocks, sample_stocks]
    weights_sample = last_weights.loc[sample_stocks]
    
    diag_vals_sample = np.maximum(np.diag(cov_sample), 0)
    std_devs_sample = np.sqrt(diag_vals_sample)
    corr_sample = cov_sample / np.outer(std_devs_sample, std_devs_sample)
    dist_sample = np.sqrt((1 - corr_sample) / 2)
    dist_sample_condensed = squareform(dist_sample.values, checks=False)
    link_matrix_sample = linkage(dist_sample_condensed, method='ward')
    
    sorted_idx_sample = get_quasi_diag(link_matrix_sample, len(cov_sample))
    cov_quasi = cov_sample.iloc[sorted_idx_sample, :].iloc[:, sorted_idx_sample]
    
    # Create a figure with two subplots for the heatmaps
    fig, axs = plt.subplots(1, 2, figsize=(16, 6), dpi=500)
    # Apply BSIC style to each axis
    for ax in axs:
        bsic.apply_bsic_style(fig, ax)
    
    im0 = axs[0].imshow(cov_sample, aspect='auto', cmap='viridis')
    axs[0].set_title("Original Covariance Matrix\n(50 Random Stocks)")
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
    
    im1 = axs[1].imshow(cov_quasi, aspect='auto', cmap='viridis')
    axs[1].set_title("Quasi-diagonalized Covariance Matrix\n(Clusters Grouped)")
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    # Optionally, apply BSIC logo to the figure (choose one axis for logo)
    bsic.apply_bsic_logo(fig, axs[0])
    plt.show()
    
    # Create dendrogram figure with BSIC styling and logo
    fig2, ax2 = plt.subplots(figsize=(10, 5), dpi=500)
    bsic.apply_bsic_style(fig2, ax2)
    dendrogram(last_link_matrix, labels=['']*len(last_shrunk_cov.index), leaf_rotation=0, ax=ax2)
    ax2.set_title("Dendrogram for Last Shrunk Covariance Matrix (No Ticker Labels)")
    ax2.set_xlabel("Stock")
    ax2.set_ylabel("Distance")
    plt.tight_layout()
    bsic.apply_bsic_logo(fig2, ax2)
    plt.show()
    
    print("\nLast HRP weights for the sampled 50 stocks (in original order):")
    print(weights_sample.values)

# ===============================
# 6. Save Returns Data to CSV
# ===============================
hrp_returns = pd.DataFrame({"Cumulative Return": values_all}, index=dates_all)
hrp_returns.index.name = "Date"
hrp_returns.to_csv("hrp_portfolio_returns.csv")
print("Backtest complete. Returns saved to 'hrp_portfolio_returns.csv'.")