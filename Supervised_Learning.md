# SUPERVISED LEARNING
# DS -> ML -> SL -> Regression in Data Science

## Where It Fits
```
Data Science
│
└── Machine Learning
    │
    ├── Supervised Learning <--- YOU ARE HERE
    │   ├── Regression
    │   │   ├── Linear Regression
    │   │   ├── Polynomial Regression
    │   │   └── Ridge / Lasso
    │   │
    │   └── Classification
    │       ├── Logistic Regression
    │       └── Decision Trees
    │
    └── Unsupervised Learning
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

## Regression Types

### Linear Regression
> Fits a straight line through data. Assumes a constant relationship between each input and the output.

**When to use:** The relationship between your inputs and output is roughly linear.

**Example:** Predicting house price from size.

| Size (sqft) | Price       |
|-------------|-------------|
| 1,000       | $200,000    |
| 1,500       | $300,000    |
| 2,000       | $400,000    |
| 2,500       | $500,000    |

The model learns a straight line:

```
Price = 50,000 + (200 × Size)
```

Each additional sqft adds a flat $200. The relationship is constant across all sizes — that's the "linear" part.

---

### Polynomial Regression
> Fits a curve instead of a straight line. Adds powers of X (X², X³, etc.) to capture non-linear patterns.

**When to use:** The relationship bends — adding more of something has diminishing (or accelerating) returns.

**Example:** Predicting fuel efficiency from speed.

| Speed (mph) | MPG  |
|-------------|------|
| 20          | 22   |
| 40          | 35   |
| 60          | 32   |
| 80          | 24   |
| 100         | 14   |

Fuel efficiency peaks around 40–50 mph, then drops. A straight line can't capture this. A polynomial model can:

```
MPG = β₀ + β₁(Speed) + β₂(Speed²)
```

The Speed² term lets the curve bend. Higher-degree terms (³, ⁴) bend more, but risk overfitting.

---

### Ridge / Lasso Regression
> Both are forms of *regularized* linear regression — they add a penalty for large coefficients to prevent the model from overfitting.

**When to use:** You have many predictors, some potentially useless or highly correlated. You want to keep the model from going haywire on noisy data.

**The core idea:** Standard linear regression minimizes prediction error. Ridge and Lasso do that *plus* penalize model complexity.

| | Ridge | Lasso |
|---|---|---|
| **Penalty** | Sum of squared coefficients (β²) | Sum of absolute coefficients (\|β\|) |
| **Effect** | Shrinks all coefficients toward zero | Drives some coefficients *exactly* to zero |
| **Best for** | Many small, useful predictors | Feature selection — killing off useless ones |

**Example:** Predicting house price with 10 features, several of which are noise.

```
# Standard linear regression might learn:
Price = 50k + 150(Size) + 9k(Bedrooms) + 800(LotAge) + 200(ZipCode) + ...
#                                            ↑ useless          ↑ useless

# Lasso zeroes out the weak predictors:
Price = 50k + 148(Size) + 8.5k(Bedrooms) + 0(LotAge) + 0(ZipCode) + ...
```

Ridge keeps all predictors but shrinks the noisy ones. Lasso is more aggressive — it performs automatic feature selection by eliminating them entirely.

> **Rule of thumb:** If you suspect only a few predictors truly matter, use Lasso. If you think most predictors matter but are noisy, use Ridge.

---

## The 3 Stages Your Key Terms Belong To

### 1. Data Preprocessing
> Steps taken *before* fitting the model.

- **Dummy Variables** — Categorical variables transformed into binary variables (0 or 1) for use in regression models.
- **One-Hot Encoding** — A technique to convert categorical variables into binary variables, where each category is represented by a separate column.

---

### 2. Model Fitting
> The building blocks of the regression equation itself.

- **Dependent Variable (Y)** — The outcome variable that the model aims to predict.
- **Independent Variables (X)** — The input variables used to predict the dependent variable.
- **Intercept (β₀)** — The value of the dependent variable when all independent variables are equal to zero.
- **Coefficients (β)** — The estimated values representing the relationship between each independent variable and the dependent variable.

---

### 3. Model Evaluation
> Metrics used *after* fitting to measure how well the model did.

- **R-Squared** — The proportion of the variance in the dependent variable that is predictable from the independent variables.
- **Adjusted R-Squared** — A modified version of R-Squared that adjusts for the number of predictors, providing a more accurate measure when there are multiple predictors.
- **Mean Absolute Error (MAE)** — The average of the absolute errors between the predicted and actual values.
- **Root Mean Squared Error (RMSE)** — The square root of the average of the squared differences between predicted and actual values.

---

## Full Worked Example (Linear Regression)

**Dataset:**

| House | Size (sqft) | Bedrooms | Neighborhood | Price    |
|-------|-------------|----------|--------------|----------|
| A     | 1,500       | 3        | Suburban     | $300,000 |
| B     | 2,000       | 4        | Urban        | $450,000 |
| C     | 1,200       | 2        | Rural        | $200,000 |
| D     | 1,800       | 3        | Urban        | $400,000 |

**Dependent Variable (Y):** `Price` — the thing we're trying to predict.

**Independent Variables (X):** `Size`, `Bedrooms`, and `Neighborhood`.

**Intercept (β₀):** If size = 0, bedrooms = 0, and neighborhood = 0, the model outputs `$50,000`. That baseline value is the intercept — not always realistic, but mathematically necessary.

**Coefficients (β):** After training, the model might learn:
- β₁ (Size) = `$150 per sqft`
- β₂ (Bedrooms) = `$10,000 per bedroom`

```
Price = 50,000 + (150 × Size) + (10,000 × Bedrooms)
```

---

**Dummy Variables:** `Neighborhood` is categorical (Suburban, Urban, Rural), so it can't be used as-is. Pick one category as the *baseline* (Rural = 0) and create binary flags for the rest:

| House | is_Suburban | is_Urban |
|-------|-------------|----------|
| A     | 1           | 0        |
| B     | 0           | 1        |
| C     | 0           | 0        |
| D     | 0           | 1        |

Rural is implied when both columns = 0. This is the dummy variable approach — **k-1 columns** for k categories.

**One-Hot Encoding:** Similar idea, but you keep *all* categories as separate columns with no baseline dropped:

| House | is_Suburban | is_Urban | is_Rural |
|-------|-------------|----------|----------|
| A     | 1           | 0        | 0        |
| B     | 0           | 1        | 0        |
| C     | 0           | 0        | 1        |
| D     | 0           | 1        | 0        |

Each row has exactly one `1`. Used more often in ML models (vs. regression where you drop one to avoid multicollinearity).

---

**Now the model makes predictions:**

| House | Actual Price | Predicted Price | Error    | \|Error\| | Error²      |
|-------|-------------|-----------------|----------|-----------|-------------|
| A     | $300,000    | $320,000        | −$20,000 | $20,000   | 400,000,000 |
| B     | $450,000    | $430,000        | +$20,000 | $20,000   | 400,000,000 |
| C     | $200,000    | $180,000        | +$20,000 | $20,000   | 400,000,000 |
| D     | $400,000    | $370,000        | +$30,000 | $30,000   | 900,000,000 |

**MAE:** Average of |Error| → `(20k + 20k + 20k + 30k) / 4 = $22,500`

On average, predictions are off by $22,500.

**RMSE:** √(average of Error²) → `√(525,000,000) ≈ $22,912`

Similar to MAE here, but RMSE punishes House D's larger $30k error more heavily because of the squaring step.

---

**R-Squared:** Suppose the model explains 85% of the variation in house prices — meaning size, bedrooms, and neighborhood account for most of why prices differ. R² = **0.85**.

**Adjusted R-Squared:** If you then add a useless variable like "house ID number," plain R² might creep up to 0.86 even though the new variable adds nothing. Adjusted R² would stay near **0.85**, penalizing you for adding a predictor that didn't earn its place.

---

> **Key takeaway:** MAE and RMSE tell you *how far off* your predictions are in dollar terms. R² and Adjusted R² tell you *how much of the story* your model is explaining.
