# UNSUPERVISED LEARNING
# DS -> ML -> UL -> Clustering / Dimensionality Reduction / Anomaly Detection

## Where It Fits
```
Data Science
│
└── Machine Learning
    │
    ├── Supervised Learning
    │   ├── Regression
    │   │   ├── Linear Regression
    │   │   ├── Polynomial Regression
    │   │   └── Ridge / Lasso
    │   │
    │   └── Classification
    │       ├── Logistic Regression
    │       └── Decision Trees
    │
    └── Unsupervised Learning <--- YOU ARE HERE
        ├── Clustering
        │   ├── k-means
        │   ├── Hierarchical Clustering
        │   └── DBSCAN
        │
        ├── Dimensionality Reduction
        │   ├── PCA
        │   └── t-SNE
        │
        └── Anomaly Detection
            └── Isolation Forest
```

---

> Unlike supervised learning, there is no Y to predict. The model finds structure in the data on its own.

---

## Clustering

### k-means
> Partitions data into k groups by minimizing the distance between each point and its cluster's center.

**When to use:** You want to segment data into a known (or guessed) number of groups.

**Example:** Grouping customers by spending behavior.

| Customer | Purchases/yr | Avg Spend | Cluster     |
|----------|--------------|-----------|-------------|
| A        | 2            | $20       | Budget      |
| B        | 15           | $200      | High-Value  |
| C        | 3            | $25       | Budget      |
| D        | 12           | $180      | High-Value  |
| E        | 7            | $90       | Mid-Tier    |

You set k=3, and the algorithm iterates until it finds 3 stable center points. No one told it "Budget" or "High-Value" — it found those groupings purely from the numbers.

> **Limitation:** You have to choose k upfront, and it assumes clusters are roughly round/equal in size.

---

### Hierarchical Clustering
> Builds a tree of clusters (called a **dendrogram**) by repeatedly merging the two closest groups.

**When to use:** You don't know how many clusters you want, or you want to explore structure at multiple levels of granularity.

**Example:** Using the same customer data, the algorithm builds this merge sequence:

```
Step 1: Merge A & C           → [A,C]        (both budget spenders)
Step 2: Merge B & D           → [B,D]        (both high-value)
Step 3: Merge [A,C] & E       → [A,C,E]      (mid-to-low)
Step 4: Merge [A,C,E] & [B,D] → one cluster  (everyone)
```

You cut the tree at whatever level makes sense. Cut early → many small clusters. Cut late → a few big ones.

> **Advantage over k-means:** No need to pre-specify k. The dendrogram shows you all possible groupings at once.

---

### DBSCAN
> Groups points that are densely packed together, and flags isolated points as **outliers** — no cluster assignment at all.

**When to use:** Your clusters aren't round (they're oddly shaped), or you expect noise/outliers in the data.

**Example:** Detecting fraud in transaction locations.

```
Normal transactions:  tight clusters around home city and office
Fraud transactions:   lone points far from any cluster → flagged as noise
```

Unlike k-means, DBSCAN doesn't force every point into a cluster. Points with no dense neighbors are labeled as outliers — which is often exactly what you're looking for.

> **Key parameters:** `epsilon` (how close is "close") and `min_samples` (how many neighbors to count as dense).

---

## Dimensionality Reduction

### PCA (Principal Component Analysis)
> Finds the directions in your data that capture the most variance, and projects everything onto fewer dimensions.

**When to use:** You have many features and want to reduce them for visualization, faster training, or removing redundancy.

**Example:** A dataset with 4 correlated house features.

| House | Size | Rooms | Bathrooms | Garage Size |
|-------|------|-------|-----------|-------------|
| A     | High | High  | High      | High        |
| B     | Low  | Low   | Low       | Low         |
| C     | Med  | Med   | Med       | Med         |

These 4 features are highly correlated — big houses tend to have more of everything. PCA finds that ~90% of the variation can be captured in a single axis: **overall house scale**. You compress 4 columns into 1 or 2 without losing much information.

```
PC1 (overall scale) → explains 90% of variance
PC2 (shape ratio)   → explains 7% of variance
PC3, PC4            → noise, can be dropped
```

> **Important:** PCA components aren't directly interpretable like the original features. PC1 doesn't mean "size" — it's a weighted combination of all original features.

---

### t-SNE
> A visualization technique that maps high-dimensional data into 2D or 3D, preserving local neighborhood structure.

**When to use:** You want to *see* whether clusters exist in high-dimensional data. Not for preprocessing — purely for exploration.

**Example:** You have 100 features describing customer behavior. t-SNE squashes this into a 2D scatter plot where similar customers land near each other. If you see distinct blobs, that's evidence of real clusters — even if you can't name the axes.

| | PCA | t-SNE |
|---|---|---|
| **Goal** | Reduce dimensions for modeling | Visualize clusters in 2D/3D |
| **Output** | New features you can use downstream | A plot — don't use axes as features |
| **Linear?** | Yes | No |
| **Start here?** | ✅ Yes | After PCA |

> **Rule of thumb:** Use PCA to reduce dimensions before modeling. Use t-SNE to visualize the result afterward.

---

## Anomaly Detection

### Isolation Forest
> Detects outliers by trying to *isolate* each point using random splits. Anomalies are easier to isolate — they need fewer splits to be separated from the rest.

**When to use:** You want to flag unusual data points — fraud, equipment failure, network intrusions — without labeling examples upfront.

**Example:** Detecting a faulty sensor in a factory.

| Reading   | Temp (°C) | Pressure | Vibration | Isolated Quickly? |
|-----------|-----------|----------|-----------|-------------------|
| Normal    | 72        | 1.2      | 0.3       | No — blends in    |
| Normal    | 74        | 1.1      | 0.3       | No — blends in    |
| **Fault** | **140**   | **3.8**  | **2.1**   | **Yes — outlier** |

The faulty reading (140°C, high pressure, high vibration) sits far from the dense normal region. Isolation Forest isolates it in very few random splits and assigns it a high anomaly score.

> **Key advantage:** Works without any labeled examples of what "anomalous" looks like — purely unsupervised.

---

> **Key takeaway:** Clustering finds groups. Dimensionality reduction compresses features. Anomaly detection flags the odd ones out. All three do it without being told the answer first.
