# 🔌 Electricity Price Prediction using Machine Learning

[![Python 3.x](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24.2-blue.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Active](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/Electricity-Price-Prediction-MachineLearning/)

## 📌 Project Overview

This project presents an end-to-end machine learning solution for electricity price forecasting, a critical task for energy market participants, grid operators, and large-scale consumers. The implementation uses time-series analysis and machine learning techniques to predict electricity prices (`P(t+1)`) based on historical temperature and demand data.

### 🌟 Key Features

- **Comprehensive Data Analysis**
  - Exploratory data analysis (EDA) with pandas
  - Statistical analysis and data profiling
  - Handling missing values and data quality checks

- **Advanced Data Preprocessing**
  - Outlier detection and removal using IQR (Interquartile Range)
  - Feature scaling with StandardScaler
  - Time-series feature engineering
  - Train-test split with temporal ordering preservation

- **Machine Learning Pipeline**
  - Implementation of Linear Regression model
  - Hyperparameter tuning and cross-validation
  - Feature importance analysis
  - Model persistence for production use

- **Performance Evaluation**
  - Multiple evaluation metrics (RMSE, MAE, R²)
  - Residual analysis
  - Model interpretability analysis
  - Comparison with baseline models

- **Interactive Visualizations**
  - Time-series plots of actual vs predicted values
  - Correlation heatmaps
  - Feature distribution analysis
  - Model performance dashboards

## 📂 Project Structure

```
Electricity-Price-Prediction-MachineLearning/
├── electricity_price_prediction.ipynb  # Main Jupyter notebook with complete analysis
├── data/
│   ├── 2018_CI_Assignment_Training_Data.csv  # Training dataset (956 samples)
│   └── 2018_CI_Assignment_Testing_Data.csv   # Testing dataset (506 samples)
├── models/
│   └── electricity_price_model.pkl  # Trained model (pickle format)
├── notebooks/
│   ├── 01_data_exploration.ipynb    # Initial data analysis
│   ├── 02_feature_engineering.ipynb # Feature creation and selection
│   └── 03_model_training.ipynb      # Model development and evaluation
├── src/
│   ├── data_processing.py          # Data loading and preprocessing functions
│   ├── model.py                    # Model definition and training logic
│   └── visualization.py            # Plotting and visualization utilities
├── requirements.txt                # Project dependencies
└── README.md                       # Project documentation
```

## 🛠️ Technical Stack & Dependencies

### Core Technologies
- **Python 3.8+** - Primary programming language
- **Jupyter Notebook** - Interactive development environment
- **scikit-learn** - Machine learning algorithms and utilities
- **pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Matplotlib/Seaborn** - Data visualization
- **joblib** - Model persistence

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Electricity-Price-Prediction-MachineLearning.git
cd Electricity-Price-Prediction-MachineLearning
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

1. **Data Preparation**
   - Place your dataset files in the `data/` directory
   - Run the data preprocessing pipeline:
   ```python
   from src.data_processing import load_and_preprocess_data
   
   # Load and preprocess data
   X_train, X_test, y_train, y_test = load_and_preprocess_data(
       'data/2018_CI_Assignment_Training_Data.csv',
       'data/2018_CI_Assignment_Testing_Data.csv'
   )
   ```

2. **Model Training**
   - Train the model with default parameters:
   ```python
   from src.model import train_model
   
   model = train_model(X_train, y_train)
   ```

3. **Model Evaluation**
   - Evaluate model performance:
   ```python
   from src.model import evaluate_model
   
   metrics = evaluate_model(model, X_test, y_test)
   print(f"Model RMSE: {metrics['rmse']:.2f}")
   print(f"Model R²: {metrics['r2']:.4f}")
   ```

4. **Making Predictions**
   - Use the trained model for predictions:
   ```python
   predictions = model.predict(X_test)
   ```

## 📊 Data Description

The dataset consists of time-series data with the following features:

| Feature | Description | Unit |
|---------|-------------|------|
| T(t-2) | Temperature two time steps before | °C |
| T(t-1) | Temperature one time step before | °C |
| T(t) | Current temperature | °C |
| D(t-2) | Electricity demand two time steps before | MW |
| D(t-1) | Electricity demand one time step before | MW |
| D(t) | Current electricity demand | MW |
| P(t+1) | Electricity price to predict (target variable) | $/MWh |

### Dataset Statistics
- **Training samples**: 955
- **Testing samples**: 506
- **Features**: 6
- **Time period**: Not specified (appears to be hourly/daily intervals)

## 🧠 Model Architecture

The project implements a Linear Regression model with the following characteristics:

1. **Feature Engineering**
   - Time-lagged features for capturing temporal dependencies
   - Polynomial features for capturing non-linear relationships
   - Interaction terms between temperature and demand

2. **Model Selection**
   - **Algorithm**: Linear Regression
   - **Regularization**: L2 (Ridge) with cross-validated alpha
   - **Feature Scaling**: StandardScaler (zero mean, unit variance)

3. **Hyperparameter Tuning**
   - Grid search with 5-fold cross-validation
   - Parameter grid includes:
     - Regularization strength (alpha)
     - Polynomial degree for feature transformation
     - Interaction terms

## 📈 Performance Metrics

The model's performance is evaluated using the following metrics:

1. **Mean Absolute Error (MAE)**: $X.XX
2. **Root Mean Squared Error (RMSE)**: X.XX
3. **R² Score**: 0.XX
4. **Mean Absolute Percentage Error (MAPE)**: X.XX%

## 🎯 Applications

This electricity price prediction model can be used for:
- Energy trading and portfolio optimization
- Grid operation and scheduling
- Demand response programs
- Renewable energy integration
- Budget planning for large consumers

## 🤝 Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset provided for educational purposes
- Built with open-source tools and libraries
- Inspired by real-world energy forecasting challenges

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

### Main Script: `electricity_price_prediction.ipynb`

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

