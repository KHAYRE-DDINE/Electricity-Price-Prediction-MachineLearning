# Electricity Price Prediction (Fuzzy System)

## Overview
- Predicts next-interval electricity price (`P(t+1)`) for Queensland, AU using a Mamdani-type fuzzy inference system.
- Focuses on interpretability by modeling human-like reasoning over temperature and demand.
- Implemented in Python 3.6 with Jupyter Notebook.

## Project Structure
- `Forecasting Electricity Price (Fuzzy System).ipynb` — full workflow: data prep, feature analysis, fuzzy system, evaluation.
- `2018_CI_Assignment_Training_Data.csv` — training dataset.
- `2018_CI_Assignment_Testing_Data.csv` — testing dataset.
- Generated during execution:
  - `trainingDataset_clean.csv` — cleaned training data
  - `testingDataset_clean.csv` — cleaned testing data

## Purpose
- Build an interpretable model to forecast electricity prices using domain-relevant inputs:
  - Temperature history: `T(t-2)`, `T(t-1)`, `T(t)`
  - Demand history: `D(t-2)`, `D(t-1)`, `D(t)`
  - Target: `P(t+1)`

## Tools & Libraries
- Python 3.6.4 (`Forecasting Electricity Price (Fuzzy System).ipynb:1994`)
- `pandas`, `numpy`, `matplotlib` (`Forecasting Electricity Price (Fuzzy System).ipynb:40-47`)
- `scikit-fuzzy` (`skfuzzy`, `skfuzzy.control`) for fuzzy logic (`Forecasting Electricity Price (Fuzzy System).ipynb:44-46`)

## Data Preparation
- Outlier removal via IQR-based fences per dataset:
  - Training (`Q1`, `Q3`, range, index detection, deletion): `Forecasting Electricity Price (Fuzzy System).ipynb:166-181`
  - Testing: `Forecasting Electricity Price (Fuzzy System).ipynb:196-210`
- Clean datasets saved to CSV:
  - Training: `Forecasting Electricity Price (Fuzzy System).ipynb:390-399`
  - Testing: `Forecasting Electricity Price (Fuzzy System).ipynb:413-421`

## Feature Selection (Correlation Analysis)
- Correlation matrix computed over features (`A`): `Forecasting Electricity Price (Fuzzy System).ipynb:558-570`
- Observations and choices (`T(t-2)`, `D(t)`, `P(t+1)`): `Forecasting Electricity Price (Fuzzy System).ipynb:583-599`

## Fuzzy Model
- Universes:
  - Temperature: `[20, 34]` (`Forecasting Electricity Price (Fuzzy System).ipynb:765`)
  - Demand: `[3500, 7000]` (`Forecasting Electricity Price (Fuzzy System).ipynb:766`)
  - Price: `[10, 55]` (`Forecasting Electricity Price (Fuzzy System).ipynb:767`)
- Membership functions (triangular):
  - Temperature: low/medium/high (`Forecasting Electricity Price (Fuzzy System).ipynb:808-813`)
  - Demand: very low/low/medium/high/very high (`Forecasting Electricity Price (Fuzzy System).ipynb:844-850`)
  - Price: low/medium/high (`Forecasting Electricity Price (Fuzzy System).ipynb:882-886`)

## Rules
- 14 rules covering combinations of temperature and demand to price mapping:
  - Definitions: `Forecasting Electricity Price (Fuzzy System).ipynb:1749-1809`
  - Control system setup and simulation object: `Forecasting Electricity Price (Fuzzy System).ipynb:1805-1809`

## Evaluation
- Training set average relative error: `0.2299` (`Forecasting Electricity Price (Fuzzy System).ipynb:1849`)
- Testing set average relative error: `0.2292` (`Forecasting Electricity Price (Fuzzy System).ipynb:1922`)
- Plots of target vs. system output: `Forecasting Electricity Price (Fuzzy System).ipynb:1896-1904`, `Forecasting Electricity Price (Fuzzy System).ipynb:1967-1975`

## Getting Started
- Install Python and dependencies:
  - `pip install pandas numpy matplotlib scikit-fuzzy jupyter`
- Launch the notebook:
  - `jupyter notebook "Forecasting Electricity Price (Fuzzy System).ipynb"`
- Ensure the CSV files are in the same directory as the notebook.

## Usage
- Run cells sequentially:
  - Import packages and load datasets.
  - Remove outliers and regenerate clean CSVs.
  - Compute correlations and select features.
  - Define universes and membership functions.
  - Create fuzzy rules and compile the control system.
  - Evaluate performance on training and testing data.

## Extending the Model
- Adjust membership function ranges to reflect different regions or updated data.
- Modify rules to incorporate expert knowledge or additional inputs.
- Add features (e.g., time-of-day, renewable output) and retrain correlations.

## Notes
- Notebook uses a non-GUI Matplotlib backend in some environments; plots still render inline (`Forecasting Electricity Price (Fuzzy System).ipynb:788-790`).
