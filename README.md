# 🚴‍♂️ Bike Sharing Demand Prediction

This project focuses on predicting bike rental demand using a variety of machine learning and deep learning models. By analyzing hourly and daily data from a bike sharing system, we explore patterns and build models to forecast bike usage, which can be used to optimize operations and resource allocation.

---

## 📂 Dataset Description

The dataset used is the [Bike Sharing Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/bike-sharing-dataset), which contains information about bike rentals from a US-based city.

### Key Features:
- `season`, `mnth`, `hr`, `weekday`, `holiday`, `workingday`: Time-based and calendar attributes.
- `weathersit`, `temp`, `atemp`, `hum`, `windspeed`: Weather conditions.
- `casual`, `registered`, `cnt`: Count of casual, registered, and total users.
- `dateday`, `datetime`: Date and time info.

The dataset is split into two CSV files:
- `day.csv` — Aggregated daily data
- `hour.csv` — Aggregated hourly data (used for modeling)

---

## 🔍 Exploratory Data Analysis (EDA)

We performed EDA to identify trends and seasonality in the data:

- **Season vs Count** 📈
- **Yearly trends**
- **Hourly demand distribution**
- **Correlation heatmap** to find important relationships

📌 The demand is highest during business hours and in the summer/fall months.

---

## 🧠 Machine Learning Models

We trained and evaluated several regression models to predict the `cnt` (total number of rentals):

### Models Used:
- ✅ Random Forest Regressor
- ✅ Decision Tree Regressor
- ✅ K-Nearest Neighbors Regressor
- ✅ Ridge and Lasso Regressors
- 🚫 Logistic Regression *(Removed due to poor performance)*

#### 🔎 Evaluation Metrics:
- **R² Score**
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**

> ✅ **Random Forest and Decision Tree** performed best in traditional ML models.

---

## 🤖 Deep Learning Approach

We also built a **Neural Network (MLP)** using `TensorFlow` with the following architecture:

- Input Layer → Dense(128) → BatchNorm → Dropout
- Hidden Layer → Dense(64) → BatchNorm → Dropout
- Output Layer → Dense(1)

**Optimization**: RMSprop  
**Loss**: Mean Squared Error  
**Callback**: Early Stopping to avoid overfitting

📊 Evaluation:
- **R² Score**
- **MAE**
- **MSE**

We also tested **MLPRegressor from sklearn**, which gave competitive results.

---

## 🧪 Final Results (R² Score)

| Model                      | R² Score (%) |
|---------------------------|--------------|
| Random Forest Regressor   | ✅ 94.XX%     |
| Decision Tree Regressor   | ✅ 93.XX%     |
| Ridge Regressor           | ✅ 82.XX%     |
| Lasso Regressor           | ✅ 81.XX%     |
| KNN Regressor             | ✅ 78.XX%     |
| Deep Learning (TF MLP)    | ✅ 92.XX%     |
| Sklearn MLP Regressor     | ✅ 90.XX%     |

(*Exact values will vary slightly depending on random seed and train-test split.*)

---

## 🧰 Tech Stack

- **Python** 🐍
- **Pandas, NumPy, Seaborn, Matplotlib** 📊
- **Scikit-learn** 🧪
- **TensorFlow / tf.keras** 🧠

---

## 📌 Conclusion

This project demonstrates the power of regression models in forecasting real-world demand based on time, season, and weather data. Random Forest and Deep Learning models offer the most accurate predictions.

---

## 🚀 How to Run

1. **Clone this repo**:
   ```bash
   git clone https://github.com/Denistanb/Bike-Price-Prediction.git
   cd Bike-Price-Prediction
2. **Install dependencies**:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn tensorflow tensorflow-hub
3. **Run the Notebook**:
   ```bash
   jupyter notebook "Bike Price Prediction.ipynb"
