# Bike Demand Prediction

A notebook that predicts hourly bike rental counts from the UCI Bike Sharing dataset, comparing several scikit-learn regressors and a small Keras network.

## Data

`Data/hour.csv` (17,379 hourly rows) and `Data/day.csv` (731 daily rows) cover 2011 and 2012. The notebook models the hourly file, with `cnt` (total rentals) as the target.

## What the notebook does

1. Loads the daily and hourly files, tries to merge them on a datetime column, and falls back to the hourly file alone (the merge cell is left in, marked "Merging Not Working").
2. Plots rentals against season, year, month and hour, checks for missing values and draws a correlation heatmap.
3. Splits the date string into day, month and year columns, then makes an 80/20 train/test split (seed 30).
4. Trains Logistic Regression, Random Forest, KNN, Ridge, Lasso and Decision Tree models and scores each with R2, MAE and MSE.
5. Trains a Keras MLP (two dense layers with batch norm, dropout and L2, early stopping) and a scikit-learn `MLPRegressor`.

## Results and a problem with them

R2 on the test set, as printed in the notebook:

| Model | R2 |
|---|---|
| Ridge | 100.0% |
| Lasso | 100.0% |
| Random Forest | 99.98% |
| Decision Tree | 99.92% |
| KNN | 99.13% |
| Logistic Regression | -92.06% |

These scores are inflated. `casual` and `registered` stay in the feature set and they add up to `cnt`, so the models are mostly reading the answer. Dropping those two columns is the first thing to fix before trusting any of these numbers. Logistic Regression is a classifier and does not belong on a continuous target.

## Running it

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow tf-keras tensorflow-hub
jupyter notebook "Bike Demand Prediction.ipynb"
```

The notebook was written on Google Colab and reads from `/content/drive/MyDrive/Bike Sharing/`. To run it locally, change the `read_csv` paths to `Data/day.csv` and `Data/hour.csv`. The notebook also expects a `dateday` column with `dd-mm-yyyy` dates, while the CSVs in `Data/` use `dteday` with ISO dates, so the date parsing cells need adjusting too.

## Layout

```
Bike Demand Prediction.ipynb
Data/day.csv
Data/hour.csv
```
