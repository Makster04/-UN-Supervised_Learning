# ENSEMBLE METHODS
# DS -> ML -> EM -> Bagging & Boosting

## Where It Fits
```
Machine Learning
│
├── Supervised Learning
│   ├── Regression
│   │   ├── Linear Regression
│   │   ├── Polynomial Regression
│   │   └── Ridge / Lasso
│   │
│   └── Classification
│       ├── Logistic Regression
│       ├── Decision Trees
│       ├── Random Forest
│       ├── SVM
│       └── k-NN
│
├── Unsupervised Learning
│   ├── Clustering
│   │   ├── k-means
│   │   ├── Hierarchical Clustering
│   │   └── DBSCAN
│   │
│   ├── Dimensionality Reduction
│   │   ├── PCA
│   │   └── t-SNE
│   │
│   └── Anomaly Detection
│       └── Isolation Forest
│
├── Reinforcement Learning
│   ├── Q-Learning
│   └── Policy Gradient
│
└── Ensemble Methods             <--- YOU ARE HERE
    ├── Bagging
    │   └── Random Forest
    └── Boosting
        ├── XGBoost
        └── LightGBM
```

---

> Ensemble Methods combine multiple models to produce a better result than any single model alone. The core idea: weak learners that are only slightly better than random guessing, combined correctly, become a strong learner.

---

## The 3 Core Concepts

### 1. Weak vs. Strong Learners
> The foundation of why ensembles work.

- **Weak Learner** — A model that performs only slightly better than random chance (e.g. a shallow decision tree that makes one split).
- **Strong Learner** — A model with high accuracy across many scenarios.
- **Ensemble** — A collection of weak learners whose combined output becomes a strong learner.

```
One shallow tree:   60% accuracy  ← weak
100 shallow trees:  92% accuracy  ← strong (ensemble)
```

---

### 2. Bias vs. Variance
> The core tradeoff that Bagging and Boosting each solve differently.

- **Bias** — How wrong the model is on average. High bias = underfitting (too simple, misses patterns).
- **Variance** — How much the model fluctuates with different training data. High variance = overfitting (too complex, memorizes noise).
- **Bias-Variance Tradeoff** — Reducing one tends to increase the other. Ensemble methods attack this from different angles.

| | Bagging | Boosting |
|---|---|---|
| **Targets** | High variance | High bias |
| **Problem it solves** | Overfitting | Underfitting |
| **How** | Average many overfit models | Sequentially fix mistakes |

---

### 3. How Models Are Combined
> Two strategies for turning individual predictions into one final answer.

- **Voting** — Each model predicts a class; the majority wins (used in Bagging for classification).
- **Averaging** — Each model predicts a number; the average is the final answer (used in Bagging for regression).
- **Weighted Sum** — Models contribute to the final prediction in proportion to how well they performed (used in Boosting).

---

## Ensemble Method Types

### Bagging (Bootstrap Aggregating)
> Trains many models **in parallel** on random subsets of the training data, then combines their predictions by voting or averaging.

**When to use:** Your model overfits — it performs well on training data but poorly on new data. You want to reduce variance without changing the underlying algorithm.

**The core idea:** If one model overfits one quirky subset of data, that quirk gets drowned out when averaged with 99 other models trained on different subsets.

**Example:** Predicting whether a customer will churn, using 4 bagged decision trees.

Training data is randomly sampled **with replacement** (called bootstrapping) for each tree:

| Tree | Trained On         | Predicts for New Customer |
|------|--------------------|---------------------------|
| 1    | Customers A,A,B,D  | Churn                     |
| 2    | Customers B,C,C,D  | No Churn                  |
| 3    | Customers A,C,D,D  | Churn                     |
| 4    | Customers A,B,B,C  | Churn                     |

Majority vote (3 Churn, 1 No Churn) → **Final Prediction: Churn**

> **Note:** "With replacement" means the same customer can appear multiple times in one training set. This is what creates the variation between trees.

---

#### Random Forest
> Bagging applied to decision trees, with an extra twist: each tree also sees a random subset of *features* at every split — not just random rows.

**When to use:** You need a reliable, out-of-the-box model for classification or regression that handles messy data, missing values, and many features well.

**The extra twist — feature randomness:**

Without it, all trees would make the same first split on the most powerful feature (e.g. "Income"), making them highly correlated and defeating the purpose of ensembling.

```
Standard Bagging:     all trees can use all features → correlated trees
Random Forest:        each split uses a random subset of features → decorrelated trees
```

**Example:** Predicting house price. Dataset has 6 features: Size, Bedrooms, Bathrooms, Neighborhood, LotAge, GarageSize.

Each tree at each split randomly samples 3 features to consider:

```
Tree 1, Split 1: considers {Size, Bathrooms, LotAge}       → splits on Size
Tree 2, Split 1: considers {Bedrooms, Neighborhood, Size}   → splits on Neighborhood
Tree 3, Split 1: considers {GarageSize, Bathrooms, LotAge}  → splits on Bathrooms
```

No tree dominates on any one feature. The forest's average prediction is far more robust than any individual tree.

> **Key advantage:** Works well with almost no tuning. A strong default choice before trying more complex models.

---

### Boosting
> Trains models **sequentially**, where each new model focuses on correcting the mistakes of the previous ones.

**When to use:** Your model underfits — it's too simple and missing real patterns in the data. You want to reduce bias by progressively building a more accurate model.

**The core idea:** Start with a weak model. Find what it got wrong. Train the next model specifically to fix those errors. Repeat. Each model is a small correction on top of the last.

```
Model 1:  rough prediction          error = 30%
Model 2:  fixes Model 1's mistakes  error = 18%
Model 3:  fixes Model 2's mistakes  error = 11%
Model 4:  fixes Model 3's mistakes  error = 7%
...
Final:    weighted sum of all models
```

**Example:** Predicting loan default with 4 boosting rounds.

| Applicant | Actual | Round 1 | Round 2 | Round 3 | Final |
|-----------|--------|---------|---------|---------|-------|
| A         | No     | No ✅   | No ✅   | No ✅   | No ✅ |
| B         | Yes    | No ❌   | Yes ✅  | Yes ✅  | Yes ✅|
| C         | No     | No ✅   | No ✅   | No ✅   | No ✅ |
| D         | Yes    | No ❌   | No ❌   | Yes ✅  | Yes ✅|

Applicants B and D were misclassified in Round 1. Round 2 upweights them — they matter more in the next model's training. Round 3 does the same for D. By the final round, all four are correctly predicted.

---

#### XGBoost (Extreme Gradient Boosting)
> A highly optimized implementation of gradient boosting that adds regularization, handles missing values natively, and runs in parallel despite boosting being sequential.

**When to use:** Tabular data competitions, production ML pipelines, anywhere you need high accuracy fast. XGBoost has won more Kaggle competitions than any other single algorithm.

**What makes it "extreme":**

| Feature | Standard Boosting | XGBoost |
|---|---|---|
| **Regularization** | None | L1 and L2 built in (prevents overfitting) |
| **Missing values** | Must be handled manually | Learned automatically |
| **Speed** | Single-threaded | Parallelized tree building |
| **Pruning** | Grows then prunes | Prunes during growth (more efficient) |

**Example:** Predicting customer churn with 3 features.

```
Round 1 prediction (base model):
  Customer A → 0.4  (actual: 0 = No Churn)  residual = −0.4
  Customer B → 0.3  (actual: 1 = Churn)     residual = +0.7
  Customer C → 0.6  (actual: 1 = Churn)     residual = +0.4

Round 2 trains on residuals:
  Learns to add +0.5 for customers like B, −0.2 for customers like A

Final score = Round 1 output + Round 2 correction + ...
```

Each round fits a new tree to the *residuals* (the gap between current prediction and actual). XGBoost adds a penalty term to keep the trees shallow and prevent overfitting.

> **Rule of thumb:** XGBoost is usually the first "serious" model to try on any structured/tabular dataset after establishing a baseline.

---

#### LightGBM (Light Gradient Boosting Machine)
> A faster, more memory-efficient variant of gradient boosting developed by Microsoft, designed for very large datasets.

**When to use:** You have millions of rows, hundreds of features, and XGBoost is too slow or uses too much memory.

**How it's different from XGBoost:**

| | XGBoost | LightGBM |
|---|---|---|
| **Tree growth** | Level-wise (grows row by row) | Leaf-wise (grows the most useful leaf first) |
| **Speed** | Fast | Faster (up to 10× on large data) |
| **Memory** | Higher | Lower |
| **Accuracy** | High | Comparable or better on large data |
| **Small datasets** | ✅ Reliable | ⚠️ Can overfit |

**Example:** The difference in tree growth strategy.

```
Level-wise (XGBoost):         Leaf-wise (LightGBM):
        Root                          Root
       /    \                        /    \
      A      B          →           A      B
     / \    / \                          /    \
    C   D  E   F                        E      F
                                              /  \
                                             G    H
```

LightGBM keeps splitting the leaf with the highest gain rather than all leaves at the same depth. This produces deeper, more accurate trees faster — but risks overfitting on small datasets.

> **Rule of thumb:** Start with XGBoost. Switch to LightGBM if your dataset has more than ~100,000 rows and training time becomes a bottleneck.

---

### Bagging vs. Boosting — Key Differences

| | Bagging | Boosting |
|---|---|---|
| **Training order** | Parallel (independent) | Sequential (each learns from last) |
| **Focus** | Reduce variance (overfitting) | Reduce bias (underfitting) |
| **Weighting** | All models equal | Later models weighted by performance |
| **Speed** | Faster (parallelizable) | Slower (must be sequential) |
| **Sensitive to outliers** | Less | More (outliers get upweighted) |
| **Main algorithms** | Random Forest | XGBoost, LightGBM |

---

## The 3 Stages of an Ensemble Pipeline

### 1. Base Model Selection
> Choosing the weak learner each ensemble will be built from.

- **Decision Trees** — By far the most common base learner for both Bagging and Boosting. Shallow trees (depth 1–3) are the standard choice for Boosting.
- **Depth matters:** Bagging uses full-depth trees (high variance, low bias). Boosting uses shallow trees called **stumps** (low variance, high bias — corrected over rounds).

```
Bagging base learner:   deep tree   (max_depth = unlimited)
Boosting base learner:  stump       (max_depth = 1 or 2)
```

---

### 2. Training
> How models are fit and combined during the ensemble process.

- **n_estimators** — The number of trees to build. More trees = better up to a point, then diminishing returns.
- **Learning Rate (η)** — In Boosting, how much each new tree's correction is applied. Small η = slower but more stable. Large η = faster but risks overshooting.
- **Subsample** — The fraction of training data used per tree. Less than 1.0 adds randomness, reducing overfitting.

| Parameter | Bagging (Random Forest) | Boosting (XGBoost) |
|---|---|---|
| n_estimators | 100–500 | 100–1000 |
| Learning rate | N/A | 0.01–0.3 |
| Max depth | Unlimited | 3–8 |
| Subsample | 0.8–1.0 | 0.6–0.9 |

---

### 3. Evaluation
> Metrics used after training to measure how well the ensemble performs.

- **Accuracy** — Proportion of correct predictions. Good for balanced classes.
- **Precision** — Of all predicted positives, how many were actually positive? (Minimizes false alarms)
- **Recall** — Of all actual positives, how many did we catch? (Minimizes missed cases)
- **F1 Score** — Harmonic mean of Precision and Recall. Best single metric when classes are imbalanced.
- **AUC-ROC** — Measures how well the model separates classes across all thresholds. 0.5 = random, 1.0 = perfect.

---

## Full Worked Example (XGBoost — Gradient Boosting)

**Dataset:** Predicting whether a student passes an exam.

| Student | Study Hours | Sleep Hours | Passed? |
|---------|-------------|-------------|---------|
| A       | 8           | 7           | Yes     |
| B       | 2           | 5           | No      |
| C       | 5           | 8           | Yes     |
| D       | 1           | 4           | No      |

**Step 1 — Base prediction:** Start with the average outcome. 2 out of 4 passed → base prediction = 0.5 for everyone.

| Student | Actual | Prediction | Residual |
|---------|--------|------------|----------|
| A       | 1      | 0.5        | +0.5     |
| B       | 0      | 0.5        | −0.5     |
| C       | 1      | 0.5        | +0.5     |
| D       | 0      | 0.5        | −0.5     |

---

**Step 2 — Round 1 tree:** Fit a shallow tree to predict the residuals. Best split found: Study Hours > 4.

```
       Study Hours > 4?
          /         \
        Yes           No
         /               \
  Residual: +0.5    Residual: −0.5
  (Students A, C)  (Students B, D)
```

Update predictions (learning rate η = 0.5):

| Student | Old Prediction | Correction       | New Prediction |
|---------|----------------|------------------|----------------|
| A       | 0.5            | + (0.5 × +0.5)   | 0.75           |
| B       | 0.5            | + (0.5 × −0.5)   | 0.25           |
| C       | 0.5            | + (0.5 × +0.5)   | 0.75           |
| D       | 0.5            | + (0.5 × −0.5)   | 0.25           |

---

**Step 3 — Round 2 tree:** Recalculate residuals and fit another tree. Now the split might use Sleep Hours.

After Round 2, predictions tighten further toward the actual values. Each round chips away at the remaining error.

---

**Final predictions after 3 rounds:**

| Student | Actual | Final Score | Threshold (0.5) | Correct? |
|---------|--------|-------------|-----------------|----------|
| A       | Yes    | 0.89        | Pass            | ✅       |
| B       | No     | 0.11        | Fail            | ✅       |
| C       | Yes    | 0.85        | Pass            | ✅       |
| D       | No     | 0.14        | Fail            | ✅       |

All four correctly predicted. Each round made a small, targeted correction to the previous round's mistakes.

---

> **Key takeaway:** Bagging makes many independent models disagree and averages out the noise. Boosting makes each model apologize for the last one's mistakes. Both beat any single model — just for opposite reasons.
