# Customer Segmentation Analysis

Unsupervised machine learning project that segments retail mall customers into distinct groups based on their annual income and spending behavior, using K-Means clustering.

## Problem Statement
Businesses often treat their entire customer base as one uniform group, leading to inefficient marketing. This project segments customers into meaningful groups so that marketing and business decisions can be tailored to each group's needs.

## Tools & Libraries
- Python 3
- Pandas, NumPy — data handling
- Matplotlib, Seaborn — visualization
- Scikit-learn — K-Means clustering, StandardScaler, Silhouette Score

## Dataset
- `Mall_Customers.csv` — 200 customer records
- Columns: CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100)

## Approach
1. Load and clean the data
2. Scale features (Annual Income, Spending Score)
3. Determine optimal number of clusters using the Elbow Method + Silhouette Score
4. Train a K-Means model (k = 5)
5. Visualize and interpret the resulting customer segments

## Files
- `customer_segmentation.ipynb` — full notebook with code, charts, and analysis
- `customer_segmentation.py` — standalone source code, refactored into documented functions (`load_data`, `clean_data`, `select_and_scale_features`, `find_optimal_k`, `train_kmeans`, `assign_clusters`, `visualize_clusters`, `summarize_segments`, `main`)
- `Mall_Customers.csv` — dataset used
- `README.md` — this file

## How to Run

Notebook:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook customer_segmentation.ipynb
```

Script:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python customer_segmentation.py
```

## Output
Five customer segments (e.g. High Income–High Spend, Low Income–High Spend, Average Customers) with a summary profile for each, usable for targeted marketing and retention strategies.
