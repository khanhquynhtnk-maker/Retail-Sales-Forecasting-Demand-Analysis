# COMPREHENSIVE SALES FORECASTING: A COMPARATIVE STUDY OF MODEL SVR AND DECISION TREE REGRESSION

## 1. Project Overview and Task

Sales forecasting plays a central role in retail operations, directly affecting inventory turnover, stockout rates, and supply chain costs. In supermarket retail, customer demand exhibits high variance, extreme right-skewness, and localized store-product interaction effects.

This project investigates sales prediction under a constrained setting: a cross-sectional dataset of 1,730 store-item records collected on a single calendar day (January 12, 2013). Because the data contains no historical time dimension, conventional time-series approaches such as autoregressive lags or rolling averages cannot be used.

The primary objective is to evaluate two regression methods:
- Support Vector Regression (SVR) using a non-linear Radial Basis Function (RBF) kernel.
- Decision Tree Regression with complexity control.

We benchmark both models against Random Forest and Linear Regression to answer a practical business question: which model provides the most effective balance between predictive accuracy, computational efficiency, and operational interpretability for store replenishment?



## 2. Methodology and Implementation

### Data Exploration and Outlier Analysis
The dataset contains 1,730 observations and 13 attributes covering store identifiers, product categories, perishability flags, and sales quantities. Initial inspection showed 0 missing values and 0 duplicate rows.

Summary of key data properties:
- The target variable (unit_sales) is heavily right-skewed, with a mean of 6.63, a median of 4.00, and a standard deviation of 8.30. Sales values range from 0.192 to 100.535 units, with skewness equal to 3.993.
- Perishable products account for 20.23% of the inventory. They exhibit higher turnover, wider interquartile ranges, and larger demand spikes than shelf-stable goods.
- Store format analysis indicates that staple groceries (GROCERY I) dominate total sales across all store types, while secondary volume comes from beverages and cleaning goods.

Outlier detection was conducted using Tukey's boxplot rule:
- Q1 = 2.0, Q3 = 8.0, IQR = 6.0
- Upper Outlier Cutoff = Q3 + 1.5 * IQR = 17.0

This identifies 120 observations (6.94% of the dataset) with sales above 17 units. Rather than removing these records, which represent high-revenue products critical to grocery operations, we stabilized target variance using the natural logarithm:

```math
y = \ln(1 + \text{unit\_sales})

### Cross-Sectional Feature Engineering
To capture localized demand patterns without time-series features, six domain-specific variables were engineered:
1. store_family: interaction between store branch and product family (28 categories).
2. store_class: granular interaction between store branch and merchandise class (266 categories).
3. family_perishable: interaction between product family and perishable status (20 categories).
4. class_freq: occurrence count of each product class across the dataset, serving as a proxy for product popularity.
5. family_freq: occurrence count of each product family, representing general department scale.
6. product_complexity: number of unique items within each family, measuring assortment diversity.

All engineered features preserved full data completeness (0 null values) and added meaningful cross-sectional variation.

### Preprocessing and Encoding Strategy
- Categorical features (store_family, store_class, family_perishable) were transformed using One-Hot Encoding. This avoids imposing false numerical order on nominal categories and expands the feature set from 19 to 340 columns.
- Continuous numerical features (class_freq, family_freq, product_complexity, store_type) were standardized to zero mean and unit variance using StandardScaler. This prevents scale disparities from distorting Euclidean distances in the SVR RBF kernel:

$$z = \frac{x - \mu}{\sigma}$$

### Model Training and Cross-Validation
Because the data represents a single-day snapshot, model evaluation was conducted using 5-fold cross-validation with shuffling (random_state = 42). All evaluations were computed on the log scale.
- SVR: trained with an RBF kernel and epsilon-insensitive loss over a grid of C in {0.1, 1.0, 10.0, 100} and epsilon in {0.01, 0.1, 0.5} (12 configurations).
- Decision Tree: trained using Mean Squared Error criteria over a grid of max_depth in {5, 10, 15, 20, 25, 30, None} and min_samples_split in {5, 10, 20, 30, 40} (35 configurations).
- Benchmarks: evaluated against Random Forest (50 estimators) and OLS Linear Regression under identical 5-fold splits.

---

## 3. Experimental Results

### SVR Hyperparameter Sensitivity
Grid search results across 12 configurations show that regularization parameter C plays a dominant role in controlling error:
- When C is small (0.1), the model underfits the data (negative R²).
- When C is large (10.0 or 100.0), the model overfits the high-dimensional feature space, widening the generalization gap up to 0.1174.
- Setting C = 1.0 achieves the best balance between bias and variance.
- Increasing epsilon to 0.5 consistently improved generalization by allowing the regressor to ignore minor daily retail noise.

The best SVR configuration (C = 1.0, epsilon = 0.5) achieved:
- Test RMSE: 0.7502
- Test MAE: 0.6084
- Test R²: 0.0387
- Overfit Gap: 0.0313
- Training Time: 3.46 seconds

### Decision Tree Optimization and Split Interpretation
For Decision Tree Regression, unconstrained trees (max_depth = None) suffered severe overfitting (Overfit Gap = 0.1296). Enforcing depth and sample split constraints restored generalization.

The best Decision Tree configuration (max_depth = 20, min_samples_split = 40) achieved:
- Test RMSE: 0.7630
- Test MAE: 0.6145
- Test R²: 0.0045
- Overfit Gap: 0.0847
- Training Time: 0.27 seconds

Inspecting the decision splits confirms intuitive retail behavior:
- Root split: class_freq_scaled <= -0.50. The model first separates low-turnover specialty products from high-frequency staple goods.
- Subsequent branches split on perishable poultry items (family_perishable_POULTRY_1) and store-specific cleaning categories, predicting higher log sales (2.83, equivalent to ~15.9 units) for high-velocity perishable lines.

### Multi-Model Benchmark Comparison

Performance across all evaluated models under 5-fold cross-validation on the log scale:

| Model | Hyperparameter Setting | Test RMSE | Test MAE | Test R² | Overfit Gap | Training Time | Interpretability |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Support Vector Regression | C = 1.0, epsilon = 0.5 | 0.7502 | 0.6084 | 0.0387 | 0.0313 | 3.46 s | Low (kernel black-box) |
| Decision Tree Regression | max_depth = 20, min_samples_split = 40 | 0.7630 | 0.6145 | 0.0045 | 0.0847 | 0.27 s | High (explicit rules) |
| Random Forest Regressor | n_estimators = 50 | 0.7633 | 0.6129 | 0.0029 | 0.1162 | 11.76 s | Low (ensemble voting) |
| Linear Regression (OLS) | Default intercept | 0.7675 | 0.6191 | -0.0079 | 0.1248 | 1.49 s | Moderate (coefficients) |

Overall, SVR achieved the lowest numerical error, while Linear Regression performed worst due to its inability to capture non-linear demand interactions.

---

## 4. Deployment Recommendation and Business Insights

Although SVR yielded the lowest test error, Decision Tree Regression is recommended for operational retail deployment based on three practical criteria:

1. Marginal performance difference:
The accuracy advantage of SVR over Decision Tree is narrow (RMSE difference of 0.0128, a gap of under 1.7%). In production, such a minor margin rarely justifies adopting an opaque model.

2. White-box business actionability:
Decision tree splits translate directly into interpretable rules that store managers and inventory controllers can audit. Knowing that specific store-class combinations generate predictable demand allows planners to calibrate safety stock, shelf space, and order schedules directly.

3. Speed and maintenance overhead:
Decision Tree trains in 0.27 seconds, compared to 3.46 seconds for SVR and 11.76 seconds for Random Forest. When expanding to hundreds of stores and tens of thousands of SKUs, sub-second training enables automated daily retraining without expensive computing infrastructure. Furthermore, decision trees do not require continuous distance metric maintenance at inference time.

Limitations and next steps:
The modest R² across all models (under 0.04) reflects the absence of promotional flags, shelf prices, foot traffic, and multi-week historical patterns in single-day cross-sectional data. Future extensions should incorporate longitudinal transaction series and promotional covariates.


## 5. Project Credits and License

- Author: Trần Ngọc Khánh Quỳnh
- Course: Introduction to Machine Learning
- Date: March 2026
