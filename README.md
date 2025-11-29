# Electricity Price Prediction using Machine Learning

[![Python 3.x](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24.2-blue.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Active](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

## 📌 Overview

This project implements a machine learning solution for predicting electricity prices in the next time interval (`P(t+1)`) using a Jupyter notebook. The implementation uses `scikit-learn`'s `LinearRegression` model and includes comprehensive data exploration, visualization, and model evaluation. The model utilizes multiple time-lagged features including temperature and demand data to predict future electricity prices.

## 🎯 Key Features

- **Comprehensive Data Analysis** with pandas for data manipulation and exploration
- **Data Visualization** using matplotlib and seaborn for insights
- **Data Preprocessing** with outlier removal using IQR (Interquartile Range)
- **Feature Scaling** using StandardScaler for model performance
- **Model Training** with scikit-learn's Linear Regression
- **Performance Evaluation** with RMSE and R² metrics
- **Interactive Visualization** of actual vs predicted values and correlation heatmaps

## 📂 Project Structure

```
Electricity-Price-Prediction-MachineLearning/
├── electricity_price_prediction.ipynb  # Jupyter notebook with the complete analysis
├── 2018_CI_Assignment_Training_Data.csv  # Training dataset (956 samples)
├── 2018_CI_Assignment_Testing_Data.csv   # Testing dataset (506 samples)
└── README.md                            # Project documentation
```

## 🛠️ Prerequisites

- Python 3.6+
- Jupyter Notebook
- Required Python packages:
  - `pandas` - Data manipulation and analysis
  - `numpy` - Numerical operations
  - `scikit-learn` - Machine learning algorithms
  - `matplotlib` - Data visualization
  - `seaborn` - Statistical data visualization
  - `jupyter` - Interactive computing

## 🚀 Getting Started

1. Open the `electricity_price_prediction.ipynb` Jupyter Notebook
2. Run all cells to execute the analysis
3. View the results directly in the notebook

## 🎯 Project Workflow

The notebook follows a structured data science workflow:

1. **Data Loading & Initial Exploration**
   - Loads training and testing datasets
   - Displays basic statistics and data structure

2. **Data Visualization**
   - Price distribution analysis
   - Correlation heatmap of all features
   - Outlier detection and handling

3. **Data Preprocessing**
   - Outlier removal using IQR method
   - Feature scaling with StandardScaler
   - Train-test split

4. **Model Training**
   - Linear Regression model implementation
   - Training on preprocessed data

5. **Model Evaluation**
   - Performance metrics (RMSE, R²)
   - Visualization of predictions vs actual values
   - Feature importance analysis

### Features Used

The model uses the following features for prediction:
- Temperature at different time lags (T(t-2), T(t-1), T(t))
- Demand at different time lags (D(t-2), D(t-1), D(t))
- Target variable: Next time period's price (P(t+1))

## 📊 Model Performance

The current model achieves the following performance metrics:

| Metric  | Training Set | Test Set |
|---------|-------------|----------|
| RMSE    | 6.20        | 20.15    |
| R²      | 0.3951      | 0.1242   |

### Key Findings

- The model shows moderate performance on the training set but struggles to generalize to the test set
- The significant gap between training and test performance suggests potential overfitting
- Feature importance analysis can help identify the most predictive features

## 🚀 Next Steps

1. **Model Improvement**
   - Try more complex models (Random Forest, XGBoost, etc.)
   - Feature engineering to capture more complex patterns
   - Hyperparameter tuning

2. **Feature Engineering**
   - Create additional time-based features
   - Consider external factors affecting electricity prices
   - Handle seasonality and trends

3. **Deployment**
   - Convert notebook to a deployable application
   - Create an API endpoint for predictions
   - Set up automated model retraining
  - Fast training and prediction
  - Easy to interpret coefficients
  - Low computational requirements
  - Serves as an excellent baseline model

### Evaluation Metrics

Model performance is assessed using:

1. **Mean Squared Error (MSE)**:
   - Measures the average squared difference between actual and predicted values
   - Lower values indicate better performance

2. **R² Score (Coefficient of Determination)**:
   - Represents the proportion of variance in the dependent variable that's predictable
   - Ranges from 0 to 1, with 1 indicating perfect prediction

## 📊 Results Interpretation

### Performance Metrics
After running the model, you'll see output similar to:
```
Model Performance:
Mean Squared Error: [value]
R² Score: [value]
Coefficients: [value1, value2]
Intercept: [value]
```

### Visualization

1. **Actual vs Predicted Scatter Plot**:
   - Displays the relationship between actual and predicted prices
   - The red dashed line indicates perfect predictions
   - Points closer to the line indicate better model performance

2. **Time Series Prediction Plot**:
   - Shows actual and predicted prices over time
   - Helps visualize how well the model tracks price trends

## 🧩 Code Structure

### Main Script: `linear_regression_simple.py`

```python
def remove_outliers(data, col_idx=6):
    # IQR-based outlier removal
    # ...

def main():
    # 1. Data Loading
    # 2. Data Preprocessing
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
   - Removes data points outside 1.5 * IQR from the quartiles

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

## 🚀 Usage Guide

### Basic Usage

1. **Prepare Environment**
   ```bash
   # Create and activate virtual environment (recommended)
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Run the Model**
   ```bash
   python linear_regression_simple.py
   ```

3. **Interpret Results**
   - Check console output for performance metrics
   - Review generated visualizations in the plots directory

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

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset provided for educational purposes
- Built with ❤️ using Python and scikit-learn
- The original fuzzy workflow remains in `Forecasting Electricity Price (Fuzzy System).ipynb` for reference.
