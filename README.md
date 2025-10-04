# Bike Demand Prediction Project

## Project Description
This project aims to predict bike rental demand using machine learning and deep learning techniques. By analyzing historical data from a bike sharing system, the project builds models to forecast bike usage, helping optimize operations and resource allocation for bike sharing services.

---

## Project Details

### Dataset
The project uses the [Bike Sharing Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/bike-sharing-dataset), which contains daily and hourly records of bike rentals in a US city. The dataset includes features such as season, month, hour, weekday, holiday, working day, weather conditions, temperature, humidity, windspeed, and user counts (casual, registered, total).

- **day.csv**: Aggregated daily data
- **hour.csv**: Aggregated hourly data (used for modeling)

### Exploratory Data Analysis (EDA)
- Analysis of seasonal and hourly trends
- Yearly demand patterns
- Correlation analysis to identify key features
- Visualization of demand distribution

### Modeling Approaches
- **Machine Learning Models:**
  - Random Forest Regressor
  - Decision Tree Regressor
  - K-Nearest Neighbors Regressor
  - Ridge and Lasso Regressors
  - Logistic Regression (for comparison)
- **Deep Learning Model:**
  - Multi-Layer Perceptron (MLP) using TensorFlow

### Evaluation Metrics
- R² Score
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)

### Results
Random Forest, Decision Tree, and Deep Learning models provided the most accurate predictions, with R² scores above 94%.

---

## Tech Stack
- Python
- Pandas, NumPy, Seaborn, Matplotlib
- Scikit-learn
- TensorFlow / tf.keras

---

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/DCode-v05/Bike-Demand-Prediction.git
   cd Bike-Demand-Prediction
   ```
2. **Install dependencies:**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn tensorflow tensorflow-hub
   ```
3. **Run the Jupyter Notebook:**
   ```bash
   jupyter notebook "Bike Demand Prediction.ipynb"
   ```

---

## Usage
- Open the notebook `Bike Demand Prediction.ipynb` in Jupyter Notebook or JupyterLab.
- Follow the sections for data exploration, preprocessing, model training, and evaluation.
- Modify or extend the notebook to experiment with different models or parameters.

---

## Project Structure
```
Bike-Demand-Prediction/
│
├── Bike Demand Prediction.ipynb   # Main analysis and modeling notebook
├── Data/
│   ├── day.csv                   # Daily aggregated data
│   └── hour.csv                  # Hourly aggregated data
└── README.md                     # Project documentation
```

---

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your feature"
   ```
4. Push to your branch:
   ```bash
   git push origin feature/your-feature
   ```
5. Open a pull request describing your changes.

---

## Contact
- **GitHub:** [DCode-v05](https://github.com/DCode-v05)
- **Email:** denistanb05@gmail.com
