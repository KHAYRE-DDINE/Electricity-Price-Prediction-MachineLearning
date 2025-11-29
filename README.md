# Electricity Price Prediction using Machine Learning

[![Python 3.x](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Active](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

## 📌 Overview

This project implements a machine learning solution for predicting electricity prices in the next time interval (`P(t+1)`). It serves as a baseline model that replaces traditional fuzzy inference systems with a more straightforward machine learning approach using `scikit-learn`'s `LinearRegression`. The model leverages the two most influential features identified through correlation analysis: temperature with a 2-period lag (`T(t-2)`) and current electricity demand (`D(t)`).

## 🎯 Key Features

- **Simple yet effective** linear regression model for electricity price prediction
- **Data preprocessing** with outlier removal using IQR (Interquartile Range)
- **Performance evaluation** with standard regression metrics (MSE, R²)
- **Visual analytics** with comparison plots
- **Modular code structure** for easy extension and modification

## 📂 Project Structure

```
Electricity-Price-Prediction-MachineLearning/
├── linear_regression_simple.py  # Main Python script for the prediction pipeline
├── 2018_CI_Assignment_Training_Data.csv  # Training dataset
├── 2018_CI_Assignment_Testing_Data.csv   # Testing dataset
├── actual_vs_predicted.png      # Scatter plot visualization (generated)
└── predictions_over_time.png    # Time series visualization (generated)
```

## 🛠️ Prerequisites

- Python 3.6+
- Required Python packages:
  - `pandas` - Data manipulation and analysis
  - `numpy` - Numerical operations
  - `scikit-learn` - Machine learning algorithms
  - `matplotlib` - Data visualization

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/KHAYRE-DDINE/Electricity-Price-Prediction-MachineLearning.git
   cd Electricity-Price-Prediction-MachineLearning
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   (If requirements.txt doesn't exist, install packages individually: `pip install pandas numpy scikit-learn matplotlib`)

3. **Run the prediction model**
   ```bash
   python linear_regression_simple.py
   ```

4. **View results**
   - Model performance metrics will be displayed in the console
   - Visualization plots will be displayed in pop-up windows
   - Plots are also saved as PNG files in the project directory

## 🎯 Purpose & Methodology

The primary goal of this project is to predict the next time interval's electricity price (`P(t+1)`) using historical data. The model is designed to be simple yet effective, serving as a baseline for more complex forecasting systems.

### Feature Engineering

Based on correlation analysis, we've identified the following features as most predictive:

- **Input Features**:
  - `T(t-2)`: Temperature with a 2-period lag
  - `D(t)`: Current electricity demand

- **Target Variable**:
  - `P(t+1)`: Next time interval's electricity price (to be predicted)

### Data Processing Pipeline

1. **Data Loading**:
   - Training and testing datasets are loaded from CSV files
   - Data is converted to numerical arrays for processing

2. **Outlier Removal**:
   - Implements IQR (Interquartile Range) method to remove price outliers
   - Ensures model robustness against extreme values

3. **Feature-Target Split**:
   - Features (X): `[T(t-2), D(t)]`
   - Target (y): `P(t+1)`

## 🛠️ Model Architecture

### Linear Regression Model

The project employs a simple yet powerful linear regression model from `scikit-learn`:

- **Model Type**: `sklearn.linear_model.LinearRegression`
- **Training**: Standard OLS (Ordinary Least Squares) fitting
- **Advantages**:
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

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset provided for educational purposes
- Built with ❤️ using Python and scikit-learn
- The original fuzzy workflow remains in `Forecasting Electricity Price (Fuzzy System).ipynb` for reference.
