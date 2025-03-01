import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.covariance import LedoitWolf
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score
import mpl_bsic as bsic  # BSIC styling library
from mpl_bsic import apply_bsic_style, apply_bsic_logo


# For reproducibility
np.random.seed(42)

# ===============================
# 1. Load Data and Compute Returns
# ===============================
# Adjust the file name as needed
adj_close_df = pd.read_csv("adj_close_data_clean.csv", index_col=0, parse_dates=True)
returns_df = adj_close_df.pct_change().dropna()

# ===============================
# 2. Define Lookback Period and Compute Shrunk Covariance
# ===============================
# Use one-year lookback from the last available date for demonstration.
last_date = returns_df.index[-1]
one_year_ago = last_date - pd.DateOffset(years=1)
past_returns = returns_df.loc[one_year_ago:last_date].dropna()

def get_shrunk_covariance(past_returns):
    lw = LedoitWolf()
    lw.fit(past_returns)
    cov = lw.covariance_
    return pd.DataFrame(cov, index=past_returns.columns, columns=past_returns.columns)

# Compute shrunk covariance matrix for the lookback period.
shrunk_cov = get_shrunk_covariance(past_returns)

# ===============================
# 3. Compute the Distance Matrix
# ===============================
# Convert covariance to correlation, then to distance.
diag_vals = np.maximum(np.diag(shrunk_cov), 0)
std_devs = np.sqrt(diag_vals)
corr = shrunk_cov / np.outer(std_devs, std_devs)
# Distance formula: sqrt((1 - correlation) / 2)
dist = np.sqrt((1 - corr) / 2)
distance_matrix = dist.values

# Ensure the distance matrix has zeros on its diagonal and no NaN values.
distance_matrix = np.nan_to_num(distance_matrix, nan=0.0)
np.fill_diagonal(distance_matrix, 0)

# ===============================
# 4. Evaluate Different Numbers of Clusters Using Silhouette Score
# ===============================
scores = {}
for num_clusters in range(2, 11):  # try from 2 to 10 clusters
    # Perform hierarchical clustering with average linkage.
    link_matrix = linkage(squareform(distance_matrix, checks=False), method='average')
    labels = fcluster(link_matrix, num_clusters, criterion='maxclust')
    score = silhouette_score(distance_matrix, labels, metric="precomputed")
    scores[num_clusters] = score
    print(f"Number of clusters: {num_clusters}, Silhouette score: {score:.4f}")

# ===============================
# 5. Plot Silhouette Scores with BSIC Style
# ===============================
fig, ax = plt.subplots(figsize=(8, 4), dpi=500)
ax.set_title("Silhouette Scores for Different Numbers of Clusters")

bsic.apply_bsic_style(fig, ax)  # Apply BSIC styling to the figure and axis


ax.plot(list(scores.keys()), list(scores.values()), marker='o')
ax.grid(True)
bsic.apply_bsic_logo(fig, ax, location ='top right')


# Optionally, export the figure using BSIC's export function
bsic.export_figure(fig, "silhouette_scores_plot.png")

plt.show()