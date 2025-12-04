# Detailed Line-by-Line Code Explanation

This document provides a detailed breakdown of the Python code used in the `electricity_price_prediction.ipynb` notebook.

## 1. Import Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
%matplotlib inline
```

- `import numpy as np`: Imports the NumPy library, which is fundamental for scientific computing in Python, mainly for support with arrays and matrices. It is aliased as `np`.
- `import pandas as pd`: Imports the pandas library, used for data manipulation and analysis (creating DataFrames). Aliased as `pd`.
- `import matplotlib.pyplot as plt`: Imports the plotting module from Matplotlib for creating static visualizations. Aliased as `plt`.
- `import seaborn as sns`: Imports Seaborn, a statistical data visualization library based on Matplotlib that provides a high-level interface for drawing attractive graphics.
- `from sklearn.linear_model import LinearRegression`: Imports the Linear Regression algorithm from the scikit-learn library, which will be used to build the prediction model.
- `from sklearn.metrics import mean_squared_error, r2_score`: Imports metric functions to evaluate the model's performance (RMSE and R-squared).
- `from sklearn.preprocessing import StandardScaler`: Imports a tool to standardize features by removing the mean and scaling to unit variance.
- `%matplotlib inline`: A "magic command" specific to Jupyter Notebooks that ensures plots are displayed directly below the code cell that produced them.

## 2. Load and Explore Data

```python
# Load the datasets
train_data = pd.read_csv("2018_CI_Assignment_Training_Data.csv")
test_data = pd.read_csv("2018_CI_Assignment_Testing_Data.csv")

# Display first few rows
print("Training Data Head:")
display(train_data.head())

# Basic statistics
print("\nTraining Data Description:")
display(train_data.describe())
```

- `train_data = pd.read_csv(...)`: Reads the training dataset from a CSV file into a pandas DataFrame named `train_data`.
- `test_data = pd.read_csv(...)`: Reads the testing dataset from a CSV file into a pandas DataFrame named `test_data`.
- `print("Training Data Head:")`: Prints a label to the console.
- `display(train_data.head())`: Shows the first 5 rows of the training dataframe. This helps in understanding the structure of the data (columns like `T(t)`, `D(t)`, `P(t+1)`).
- `display(train_data.describe())`: Generates descriptive statistics (count, mean, std, min, max, quartiles) for each numerical column. This is crucial for spotting data distribution and potential anomalies.

## 3. Data Visualization - Price Distribution

```python
plt.figure(figsize=(10, 6))
sns.histplot(train_data.iloc[:, 6], bins=30, kde=True)
plt.title('Distribution of Electricity Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()
```

- `plt.figure(figsize=(10, 6))`: Creates a new figure for plotting with a specific size (10 inches wide, 6 inches tall).
- `sns.histplot(train_data.iloc[:, 6], bins=30, kde=True)`: Uses Seaborn to draw a histogram of the 7th column (index 6), which corresponds to the target variable `P(t+1)` (Price).
  - `bins=30`: Divides the data into 30 intervals.
  - `kde=True`: Adds a Kernel Density Estimate line (a smooth curve) over the histogram to show the distribution shape.
- `plt.title(...)`, `plt.xlabel(...)`, `plt.ylabel(...)`: Sets the title and labels for the X and Y axes.
- `plt.show()`: Renders and displays the plot.

## 4. Data Visualization - Correlation Heatmap

```python
plt.figure(figsize=(12, 8))
sns.heatmap(train_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()
```

- `plt.figure(figsize=(12, 8))`: Sets up the figure size.
- `train_data.corr()`: Calculates the correlation matrix for the dataframe, showing how every variable correlates with every other variable (ranges from -1 to 1).
- `sns.heatmap(...)`: Visualizes this matrix as a color-coded matrix.
  - `annot=True`: Writes the correlation value in each cell.
  - `cmap='coolwarm'`: Sets the color scheme (blue for negative correlation, red for positive).
  - `fmt='.2f'`: Formats the numbers to 2 decimal places.
- `plt.tight_layout()`: Automatically adjusts subplot parameters to give specified padding, ensuring labels don't overlap.

## 5. Data Preprocessing - Outlier Removal

```python
def remove_outliers(data, col_idx=6):
    """Remove outliers using IQR method"""
    q1 = np.percentile(data.iloc[:, col_idx], 25)
    q3 = np.percentile(data.iloc[:, col_idx], 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return data[(data.iloc[:, col_idx] >= lower_bound) & (data.iloc[:, col_idx] <= upper_bound)]

# Remove outliers
train_data_clean = remove_outliers(train_data)
print("Original training data size:", len(train_data))
print("Training data size after removing outliers:", len(train_data_clean))
```

- `def remove_outliers(data, col_idx=6):`: Defines a function to filter out extreme values. It defaults to checking column index 6 (the Price column).
- `q1 = np.percentile(...)`: Calculates the 25th percentile (First Quartile) of the data.
- `q3 = np.percentile(...)`: Calculates the 75th percentile (Third Quartile).
- `iqr = q3 - q1`: Calculates the Interquartile Range (IQR), a measure of statistical dispersion.
- `lower_bound` / `upper_bound`: Defines the range for "normal" data. Any data point below `Q1 - 1.5*IQR` or above `Q3 + 1.5*IQR` is considered an outlier.
- `return data[...]`: Returns a filtered version of the dataframe containing only rows where the value is within the bounds.
- `train_data_clean = ...`: Applies this function to the training data.
- `print(...)`: Displays the number of rows before and after cleaning to show how many outliers were removed.

## 6. Prepare Features and Target

```python
# Prepare features and target
X_train = train_data_clean.iloc[:, :-1]  # All columns except the last one
X_test = test_data.iloc[:, :-1]
y_train = train_data_clean.iloc[:, -1]   # Last column is the target
y_test = test_data.iloc[:, -1]

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

- `X_train = train_data_clean.iloc[:, :-1]`: Selects all rows and all columns _except_ the last one to be the input features (Temperature, Demand, etc.).
- `y_train = train_data_clean.iloc[:, -1]`: Selects all rows and _only_ the last column to be the target variable (Price).
- The same splitting is done for `X_test` and `y_test`.
- `scaler = StandardScaler()`: Initializes the scaler object.
- `X_train_scaled = scaler.fit_transform(X_train)`: Fits the scaler to the training data (calculates mean and std dev) and then transforms (scales) the training data.
- `X_test_scaled = scaler.transform(X_test)`: Transforms the test data using the _same_ mean and std dev calculated from the training set. This ensures consistency.

## 7. Train Linear Regression Model

```python
model = LinearRegression()
model.fit(X_train_scaled, y_train)
print("Model training complete")
```

- `model = LinearRegression()`: Initializes a Linear Regression model object.
- `model.fit(X_train_scaled, y_train)`: Trains the model using the scaled training features (`X`) and the target values (`y`). The model learns the coefficients (weights) for the linear equation.

## 8. Model Evaluation

```python
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Calculate metrics
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\nModel Performance:")
print(f"Train RMSE: {train_rmse:.2f}")
print(f"Test RMSE: {test_rmse:.2f}")
print(f"Train R² Score: {train_r2:.4f}")
print(f"Test R² Score: {test_r2:.4f}")
```

- `model.predict(...)`: Uses the trained model to generate price predictions for both the training and testing datasets.
- `mean_squared_error(...)`: Calculates the average squared difference between actual and predicted values.
- `np.sqrt(...)`: Takes the square root of the MSE to get the Root Mean Squared Error (RMSE), which is in the same units as the target variable (Price).
- `r2_score(...)`: Calculates the R-squared score, representing the proportion of variance in the dependent variable explained by the model.
- `print(...)`: Outputs the calculated metrics to assess how well the model performed.

## 9. Methodology Details

Based on the code and documentation in your project, the Machine Learning method used is **Linear Regression**.

Here are the specific details of the implementation:

### 1. The Algorithm

- **Model:** `LinearRegression` from the **scikit-learn** library.
- **Type:** It is a **Supervised Learning** regression algorithm.
- **Goal:** It fits a linear equation to the observed data to predict the numerical value of the electricity price.

### 2. Why this method?

- **Baseline:** Linear Regression is often used as a first-step "baseline" model to establish a benchmark for performance.
- **Interpretability:** It is easy to understand how each feature (like Temperature or Demand) affects the final Price (positive or negative correlation).
- **Speed:** It is computationally very fast to train compared to complex models like Neural Networks.

### 3. Key Techniques Used with the Model

To make the Linear Regression work better, the code applies these preprocessing techniques:

- **Feature Scaling:** Uses `StandardScaler` to normalize the data (making mean=0 and variance=1). This is crucial for Linear Regression to prevent features with large numbers (like Demand ~5000) from dominating features with small numbers (like Temperature ~25).
- **Outlier Removal:** Uses the **IQR (Interquartile Range)** method to remove extreme data points that could skew the regression line.

### 4. Input & Output

- **Input Features ($X$):**
  - Temperature at times $t$, $t-1$, $t-2$.
  - Demand (Load) at times $t$, $t-1$, $t-2$.
- **Target Output ($y$):**
  - Electricity Price at time $t+1$.
