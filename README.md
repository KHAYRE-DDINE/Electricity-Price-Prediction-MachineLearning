# Electricity Price Forecasting

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-1.4+-orange.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24+-green.svg)

A machine learning project for forecasting electricity prices using historical temperature and demand data. This project was developed as part of the academic curriculum at the Faculty of Sciences and Techniques of Al Hoceima, Abdelmalek Essaâdi University.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Team](#team)
- [License](#license)

## Project Overview

This project focuses on predicting electricity prices using machine learning techniques. The model uses historical temperature and electricity demand data to forecast future electricity prices, which is crucial for energy market participants.

### Key Features
- Data preprocessing and cleaning
- Feature engineering
- XGBoost-based prediction model
- Model evaluation and visualization
- Comprehensive documentation

## Features

- **Data Preprocessing**
  - Outlier detection and removal
  - Feature scaling
  - Handling missing values

- **Feature Engineering**
  - Temporal features
  - Statistical features (mean, std, etc.)
  - Trend analysis

- **Modeling**
  - XGBoost Regressor
  - Hyperparameter tuning
  - Cross-validation

- **Evaluation**
  - R² score
  - RMSE (Root Mean Square Error)
  - MAE (Mean Absolute Error)
  - Feature importance analysis

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/electricity-price-forecasting.git
cd electricity-price-forecasting
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. **Data Preparation**
   - Place your data files in the `data/raw/` directory
   - Expected format: CSV files with temperature, demand, and price columns

2. **Running the Model**
   - Execute the main script:
   ```bash
   python src/main.py
   ```
   - Or use the Jupyter notebook:
   ```bash
   jupyter notebook notebooks/analysis.ipynb
   ```

3. **Model Training and Evaluation**
   - The script will:
     - Preprocess the data
     - Train the XGBoost model
     - Evaluate performance
     - Generate visualizations

## Project Structure

```
electricity-price-forecasting/
├── data/
│   ├── raw/           # Raw data files
│   └── processed/     # Processed data
├── notebooks/         # Jupyter notebooks for analysis
├── src/               # Source code
│   ├── features/      # Feature engineering
│   ├── models/        # Model definitions
│   ├── utils/         # Utility functions
│   └── main.py        # Main script
├── results/           # Model outputs and visualizations
├── tests/             # Unit tests
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## Results

### Model Performance
| Metric | Training Set | Test Set |
|--------|--------------|----------|
| R²     | 0.4491       | 0.1464   |
| RMSE   | 9.10 $/MWh   | 19.90 $/MWh |
| MAE    | 6.78 $/MWh   | 15.23 $/MWh |

### Feature Importance
1. D_mean (0.32) - Average demand
2. D(t) (0.28) - Current demand
3. D_trend (0.18) - Demand trend
4. T_mean (0.12) - Average temperature
5. D(t-1) (0.10) - Previous demand

## Team

This project was developed by:

- **Aharrar Khayreddine**
- **Dahhou Houda**
- **Afallah Bilal**

**Supervisor:** Ms. Hayat Routaib

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Faculty of Sciences and Techniques of Al Hoceima
- Abdelmalek Essaâdi University
- Open-source community for their valuable libraries and tools

---

**Last Updated:** December 2025
