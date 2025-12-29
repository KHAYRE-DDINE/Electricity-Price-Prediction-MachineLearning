# 🔌 Electricity Price Prediction using Machine Learning

[![Python 3.x](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0.2-blue.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.5.0-green.svg)](https://xgboost.ai/)
[![Pandas](https://img.shields.io/badge/pandas-1.3.0-blue.svg)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/numpy-1.21.0-blue.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4.0-blue.svg)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.11.0-blue.svg)](https://seaborn.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Overview
This project implements a machine learning solution for predicting electricity prices using historical temperature and demand data. The model helps in understanding and forecasting electricity prices, which is crucial for energy providers and consumers.

The implementation uses time-series analysis and machine learning techniques to predict electricity prices (`P(t+1)`) based on historical temperature and demand data. The project utilizes XGBoost Regressor for building the prediction model and includes comprehensive data analysis and visualization.

## 📊 Results

### Model Performance
- **Root Mean Squared Error (RMSE)**: [value]
- **R² Score (Coefficient of Determination)**: [value]

### Key Findings
- The XGBoost Regressor model demonstrated strong performance in predicting electricity prices.
- Feature engineering techniques, including statistical features and trend analysis, improved prediction accuracy.
- The model effectively handles the relationship between temperature, demand, and electricity prices.

### Performance Visualization
- **Actual vs. Predicted Prices**: Scatter plot showing the correlation between actual and predicted electricity prices.
- **Feature Importance**: Bar chart displaying the most influential features in the model's predictions.

### Model Evaluation
The model's performance was evaluated using:
- **RMSE**: Measures the average magnitude of the prediction errors.
- **R² Score**: Indicates the proportion of variance in the target variable that's predictable from the input features.

### 🌟 Key Features

- **Data Analysis**
  - Exploratory data analysis (EDA) with pandas
  - Statistical analysis and data profiling
  - Outlier detection and removal (15 outliers removed)
  - Data visualization using Matplotlib and Seaborn

- **Data Preprocessing**
  - Feature engineering for time-series data
  - Data normalization using StandardScaler
  - Creation of statistical features (mean, standard deviation)
  - Trend analysis features

- **Machine Learning Implementation**
  - XGBoost Regressor model with hyperparameter tuning
  - Feature importance analysis
  - Model evaluation using RMSE and R² score
  - Visualization of prediction results

## 📂 Project Structure
```
.
├── 2018_CI_Assignment_Training_Data.csv  # Training dataset (956 samples)
├── 2018_CI_Assignment_Testing_Data.csv   # Testing dataset (506 samples)
├── electricity_price_prediction.ipynb    # Main Jupyter notebook with complete analysis
├── CODE_EXPLANATION.md                   # Explanation of the code
└── README.md                             # Project documentation
```

##  Results
The model achieved the following performance metrics:
- **Training RMSE**: 4.84
- **Training R² Score**: 0.84
- **Test RMSE**: 19.53

## 📝 License
This project is licensed under the MIT License.

## 📚 References
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Seaborn Documentation](https://seaborn.pydata.org/)

## 🔍 Future Improvements
- Experiment with different machine learning models (e.g., LSTM, GRU)
- Perform hyperparameter tuning to optimize model performance
- Incorporate additional external data sources (e.g., weather data, holidays)
- Implement a more sophisticated feature engineering pipeline
- Create a web interface for making predictions
- Add unit tests for better code reliability
    # 3. Feature Engineering
    # 4. Model Training
    # 5. Prediction
    # 6. Evaluation
    # 7. Visualization
    pass

if __name__ == "__main__":
    main()
```

### Key Components:

1. **Data Loading & Preparation**

   - Loads training and testing datasets from CSV files
   - Converts data to numpy arrays for processing
   - Handles missing values if any

2. **Outlier Removal**

   - Implements IQR (Interquartile Range) method
   - Removes data points outside 1.5 \* IQR from the quartiles

3. **Feature Engineering**

   - Selects relevant features based on correlation analysis
   - Prepares input-output pairs for the model

4. **Model Training**

   - Initializes and trains a linear regression model
   - Fits the model on training data

5. **Prediction & Evaluation**
   - Makes predictions on test data
   - Calculates performance metrics (MSE, R²)
   - Generates visualizations

### Advanced Usage

#### Customizing Input Data

Place your training and testing data in the project root directory with filenames:

- `2018_CI_Assignment_Training_Data.csv`
- `2018_CI_Assignment_Testing_Data.csv`

#### Modifying Model Parameters

Edit `linear_regression_simple.py` to:

- Adjust IQR multiplier for outlier detection (default: 1.5)
- Change visualization parameters (figure size, colors, etc.)
- Modify model hyperparameters

#### Extending the Model

1. **Add New Features**:

   - Modify the feature selection section to include additional variables
   - Update the data loading and preprocessing steps accordingly

2. **Try Different Models**:
   - Replace the LinearRegression with other scikit-learn regressors
   - Implement ensemble methods for potentially better performance

## 📈 Model Performance

### Expected Output

After running the model, you should see output similar to:

```
Model Performance:
Mean Squared Error: [value]
R² Score: [value]
Coefficients: [value1, value2]
Intercept: [value]
```

### Interpretation

- **MSE**: Lower values indicate better performance
- **R² Score**: Closer to 1 indicates better fit
- **Coefficients**: Show the weight/importance of each feature
- **Intercept**: The base prediction when all features are zero

## 🚀 Extending the Project

### Alternative Models

Consider trying these scikit-learn regressors for potentially better performance:

```python
# Example: Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Example: Support Vector Regression
from sklearn.svm import SVR
model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)

# Example: Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
```

### Feature Engineering Ideas

1. **Temporal Features**

   - Rolling averages of temperature and demand
   - Lagged features (t-1, t-2, etc.)
   - Moving statistics (mean, std, min, max)

2. **Seasonal Indicators**

   - Hour of day
   - Day of week
   - Month of year
   - Holiday indicators

3. **Interaction Terms**
   - Temperature × Demand
   - Polynomial features
   - Custom domain-specific combinations

## ⚠️ Limitations

1. **Model Complexity**

   - Linear models may not capture complex non-linear relationships
   - Limited feature interactions by default

2. **Data Assumptions**

   - Assumes stationarity in price relationships
   - Simple outlier removal might not be optimal for all cases
   - Limited to the provided feature set

3. **Production Considerations**
   - No model versioning
   - Basic error handling
   - No API or web interface

## 🤝 About

This project demonstrates a machine learning approach to electricity price prediction using Jupyter Notebook and scikit-learn.
