# Bike Demand Prediction

**Forecasting hourly bike-sharing rental counts from the UCI Bike Sharing dataset — a regression study that lines up six classic ML models against a Keras deep-learning model.**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white) ![pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white) ![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=flat&logo=googlecolab&logoColor=white)

## Overview

This project predicts how many bikes get rented in a given hour from a public bike-sharing system. It uses the [UCI / Kaggle Bike Sharing dataset](https://www.kaggle.com/datasets/lakshmi25npathi/bike-sharing-dataset) — two years of records (2011–2012) covering season, weather, temperature, humidity, wind, day-of-week, holiday and working-day flags. The goal is to take those conditions and estimate the total rental count (`cnt`).

It's a single Jupyter notebook, written and run on Google Colab, that walks the whole flow end to end: load the data, explore it, clean it, then train and compare a spread of regression models. The notebook is split into two halves — a classic machine-learning section that benchmarks six scikit-learn models, and a deep-learning section that builds a small neural network with Keras (plus a scikit-learn `MLPRegressor` for comparison). Everything is scored with the same three metrics so the models can be ranked side by side.

This is a learning-stage capstone, not a production model. It covers a lot of ground in one notebook, and the honest write-up of its limitations is in the [Notes](#notes) section below — including a target-leakage issue that makes the headline scores look better than they really are.

## Key Features

- Loads both the daily (`day.csv`, 731 rows) and hourly (`hour.csv`, 17,379 rows) views of the dataset; the hourly data is what's actually modeled.
- Parses the date string into separate day / month / year integer columns and drops the original text date.
- Exploratory data analysis with matplotlib and seaborn: rental count broken down by season, year, month, and hour of day.
- Missing-value checks with `isna().sum()` and a seaborn null-heatmap (the dataset turns out to be clean).
- Full correlation matrix rendered as an annotated 15×15 heatmap to spot which features move with demand.
- Six scikit-learn models trained on an 80/20 split: Logistic Regression, Random Forest, K-Nearest Neighbors, Ridge, Lasso, and Decision Tree.
- A scoreboard step that collects every model's R² into a dictionary, drops the one with a negative score (Logistic Regression), and plots the rest as a labeled bar chart of accuracy percentages.
- A deep-learning section using `tf_keras`: a `StandardScaler` feature-scaling step feeding a Sequential MLP with batch normalization, dropout, and L2 regularization, trained with early stopping.
- A second neural model via scikit-learn's `MLPRegressor` as a quick comparison point.
- Consistent evaluation across all models with R², Mean Absolute Error, and Mean Squared Error.

## How It Works

### 1. Data loading and merging

The notebook reads `day.csv` and `hour.csv` with pandas. It first tries to merge the two on a constructed `datetime` column (daily date plus hour offset), de-duplicates, and strips the `_y` suffix columns that come out of the join. That merge path didn't end up being useful, so the notebook falls back to working with `hour.csv` on its own — 17,379 hourly records is the working dataset.

### 2. Preprocessing and EDA

The date field is split into integer `day`, `month`, and `year` columns with a small parsing loop, and the original date column is dropped. From there the notebook checks for missing values (none meaningful), and visualizes demand against the obvious drivers:

- **Season vs count** and **Year vs count** — overall volume and the year-over-year jump from 2011 to 2012.
- **Month vs count** — the seasonal hump through the warmer months.
- **Hour vs count** — the commuter peaks you'd expect.
- A full **correlation heatmap** over every numeric column.

### 3. Machine-learning models

Features are everything except the target: `X = df.drop("cnt")`, `Y = df["cnt"]`. The data is split 80/20 with `train_test_split` (seed fixed at 30 for repeatability), giving 13,903 training rows and 3,476 test rows. Six models are then trained, each in its own cell:

| Model | scikit-learn class |
|---|---|
| Logistic Regression | `LogisticRegression` |
| Random Forest | `RandomForestRegressor` |
| K-Nearest Neighbors | `KNeighborsRegressor` |
| Ridge | `Ridge` |
| Lasso | `Lasso` |
| Decision Tree | `DecisionTreeRegressor` |

Each model predicts on the test set, then gets scored with R², MAE, and MSE. The scores are gathered into a single dictionary and DataFrame. Logistic Regression — a classifier being misused on a continuous target — produces a negative R² and is dropped, and the remaining five are plotted as a bar chart with the accuracy printed on top of each bar.

### 4. Deep-learning model

The DL half scales the features with `StandardScaler` and builds a `tf_keras` Sequential network:

- **Input layer:** Dense(128), ReLU, L2(0.001) → BatchNormalization → Dropout(0.5)
- **Hidden layer:** Dense(64), ReLU, L2(0.001) → BatchNormalization → Dropout(0.5)
- **Output layer:** Dense(1), sigmoid
- **Compile:** MSE loss, RMSprop optimizer
- **Training:** up to 100 epochs, batch size 32, 20% validation split, with `EarlyStopping` (patience 10) watching `val_loss`

After the Keras model, the notebook also fits a scikit-learn `MLPRegressor` on the same data as a second neural baseline, scored with the same R² / MAE / MSE trio.

## Results / Highlights

These are the R² scores the notebook actually produced on the hourly test set:

| Model | R² |
|---|---|
| Ridge | 100.0% |
| Lasso | 100.0% |
| `MLPRegressor` (sklearn) | 99.99% |
| Random Forest | 99.98% |
| Decision Tree | 99.92% |
| K-Nearest Neighbors | 99.13% |
| Keras MLP (sigmoid output) | 0.0% |
| Logistic Regression | -92.06% |

Logistic Regression scores a negative R² (it's a classifier, not a regressor) and is excluded from the final bar chart, which ranks the surviving models by accuracy.

A caveat worth stating plainly: most of these scores are inflated by a leakage bug. The `casual` and `registered` columns are left in the feature set, and those two add up exactly to the target `cnt`. With the answer effectively present in the inputs, the linear models in particular post near-perfect R² values, so the headline numbers overstate real predictive skill. The one model that *doesn't* score high is the Keras MLP — its `sigmoid` output caps predictions at 1 while the target runs into the hundreds, so it never learns the scale and lands at R² 0.0. See [Notes](#notes).

## Tech Stack

- **Language:** Python (delivered as a Jupyter notebook)
- **Data / analysis:** pandas, NumPy
- **Visualization:** matplotlib, seaborn
- **Machine learning:** scikit-learn — `RandomForestRegressor`, `DecisionTreeRegressor`, `KNeighborsRegressor`, `Ridge`, `Lasso`, `LogisticRegression`, `MLPRegressor`, `StandardScaler`, `train_test_split`, plus the `r2_score` / `mean_absolute_error` / `mean_squared_error` metrics
- **Deep learning:** TensorFlow, `tf_keras`, `tensorflow_hub`
- **Environment:** Google Colab (Drive-mounted data paths)

## Getting Started

### Prerequisites
- Python 3.x and Jupyter Notebook or JupyterLab (or just open it in Google Colab)
- The data files in `Data/` (`day.csv`, `hour.csv`) — both ship with this repo

### Installation
```bash
git clone https://github.com/DCode-v05/Bike-Demand-Prediction.git
cd Bike-Demand-Prediction
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow tf-keras tensorflow-hub
```

### Running
```bash
jupyter notebook "Bike Demand Prediction.ipynb"
```

The notebook was originally written on Colab and reads the CSVs from a Google Drive path (`/content/drive/MyDrive/Bike Sharing/...`). To run it locally, point those `pd.read_csv` calls at the `Data/` folder in this repo instead — for example `pd.read_csv("Data/hour.csv")`.

## Usage

Run the cells top to bottom. The notebook is laid out in clear sections — data preparation, EDA, modeling, prediction, and evaluation for the ML half, then the same flow for the deep-learning half. Each model is its own block, so you can swap in different hyperparameters, try other estimators, or rework the feature set and re-run just the parts you changed. The final bar chart gives you the model ranking once everything has been scored.

## Project Structure

```
Bike-Demand-Prediction/
├── Bike Demand Prediction.ipynb   # The whole project: EDA, 6 ML models, a Keras MLP, and evaluation (80 cells)
├── Data/
│   ├── day.csv                    # Daily aggregated rentals — 731 rows (2011–2012)
│   └── hour.csv                   # Hourly rentals — 17,379 rows; this is what gets modeled
└── README.md                      # Original project documentation
```

---

## Contact

**Portfolio:** [Denistan](https://www.denistan.me)<br>
**LinkedIn:** [Denistan](https://www.linkedin.com/in/denistanb)<br>
**GitHub:** [DCode-v05](https://github.com/DCode-v05)<br>
**LeetCode:** [Denistan_B](https://leetcode.com/u/Denistan_B)<br>
**Email:** [denistanb05@gmail.com](mailto:denistanb05@gmail.com)

Made with ❤️ by **Denistan B**
