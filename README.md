# House Price Prediction

A hands-on machine learning case study that explores the Boston Housing dataset and builds a regression model to estimate the median value of owner-occupied homes (`MEDV`) from neighborhood and housing characteristics.

The project is implemented as a Jupyter Notebook and is intended for learning and experimentation with exploratory data analysis, feature relationships, preprocessing pipelines, regression modeling, and model evaluation.

## Project contents

```text
.
├── House Price Prediction.ipynb   # End-to-end analysis and modeling notebook
├── datas.csv                      # Housing observations used by the notebook
└── housing.names                  # Dataset description and feature definitions
```

## Dataset

The dataset describes housing values in Boston-area suburbs. The CSV contains 505 observations and 14 columns:

| Feature | Description |
| --- | --- |
| `CRIM` | Per-capita crime rate by town |
| `ZN` | Proportion of residential land zoned for lots over 25,000 sq. ft. |
| `INDUS` | Proportion of non-retail business acres per town |
| `CHAS` | Charles River proximity indicator (`1` if the tract bounds the river, otherwise `0`) |
| `NOX` | Nitric oxide concentration |
| `RM` | Average number of rooms per dwelling |
| `AGE` | Proportion of owner-occupied units built before 1940 |
| `DIS` | Weighted distance to five Boston employment centers |
| `RAD` | Accessibility index for radial highways |
| `TAX` | Property-tax rate per $10,000 |
| `PTRATIO` | Pupil-teacher ratio by town |
| `B` | Transformed demographic variable supplied by the source dataset |
| `LSTAT` | Percentage of lower-status population |
| `MEDV` | Median home value in thousands of dollars; regression target |

The accompanying `housing.names` file documents 506 instances in the original dataset. The checked-in `datas.csv` currently contains 505 data rows, and the notebook identifies missing values in `RM` that are handled during preprocessing.

## Analysis workflow

The notebook follows this sequence:

1. Load `datas.csv` with pandas.
2. Inspect the data using `head()`, `info()`, and `describe()`.
3. Visualize feature distributions with histograms.
4. Create both random and stratified train/test splits. The stratified split preserves the distribution of `CHAS`.
5. Explore correlations with `MEDV` and visualize relationships such as `RM` vs. `MEDV`.
6. Engineer a `TAXRM` feature using `TAX / RM` and examine its relationship with the target.
7. Preprocess numerical data with a scikit-learn pipeline that:
   - Replaces missing values using median imputation.
   - Standardizes features with `StandardScaler`.
8. Train a `DecisionTreeRegressor`.
9. Evaluate the model using training RMSE, 10-fold cross-validation, and a held-out test set.

## Main findings from the notebook

The exploratory analysis reports the following strongest linear correlations with `MEDV`:

- Positive: `RM` (`0.6939`), average rooms per dwelling.
- Negative: `LSTAT` (`-0.7382`), lower-status population percentage.
- Other notable negative relationships include `PTRATIO`, `INDUS`, `TAX`, and `NOX`.
- The engineered `TAXRM` feature has a reported correlation of approximately `-0.5380` with `MEDV`.

Using the notebook's configured decision-tree workflow:

- Training RMSE is `0.0`, which indicates that the tree fits the training data exactly and may be overfitting.
- Ten-fold cross-validation produces a mean RMSE of approximately `4.126` with a standard deviation of approximately `0.552`.
- The final held-out evaluation reports an RMSE of approximately `4.121`.

RMSE is measured in the same units as `MEDV`, which represents home value in thousands of dollars.

## Requirements

The notebook uses Python and the following packages:

- pandas
- NumPy
- Matplotlib
- scikit-learn
- Jupyter Notebook or JupyterLab

## Getting started

Clone the repository and create an environment:

```bash
git clone https://github.com/PrachiSukreSS/CaseStudy-HousePricePrediction.git
cd CaseStudy-HousePricePrediction

python -m venv .venv
```

Activate the environment:

```bash
# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install pandas numpy matplotlib scikit-learn jupyter
```

Launch the notebook:

```bash
jupyter notebook "House Price Prediction.ipynb"
```

Then run the cells from top to bottom. The notebook expects `datas.csv` to remain in the same directory as the notebook.

## Model notes

The notebook compares a `DecisionTreeRegressor` workflow with a commented-out `LinearRegression` import. The active model is the decision tree. Cross-validation is used to provide a more realistic estimate than the training-set RMSE alone.

For future iterations, consider:

- Tuning tree depth and leaf-size parameters.
- Comparing linear regression, random forest, gradient boosting, and other regressors.
- Using a single pipeline that includes feature engineering and the estimator.
- Reporting additional metrics such as MAE and $R^2$.
- Checking the data split and preprocessing order carefully to avoid leakage.
- Adding reproducible dependency versions in a `requirements.txt` file.
- Separating notebook code into reusable Python modules and adding automated tests.

## Data and usage considerations

This repository is an educational case study rather than a production valuation system. The Boston Housing dataset is historical and has known limitations, including sensitive demographic-related variables and a target that may not represent current housing markets. Predictions should not be used for real-world lending, appraisal, or policy decisions without substantial additional validation and fairness review.

## License

No license file is currently included. Unless a license is added, reuse and redistribution are subject to the repository owner's rights.
