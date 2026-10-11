# BigMart Sales Prediction

## Project Overview

This project uses machine learning to predict sales for individual products at BigMart outlets. It explores product characteristics, outlet information, and historical sales records to build and evaluate regression models.

The project demonstrates a complete introductory machine learning workflow, from understanding and cleaning data to training models and generating predictions for a test dataset.

## Objectives

- Understand and explore the BigMart dataset.
- Clean missing values and standardize categorical values.
- Analyze sales patterns through exploratory data analysis (EDA).
- Build and compare regression models.
- Tune model hyperparameters to improve validation performance.
- Generate sales predictions for the test dataset.

## Dataset

The project uses the Kaggle BigMart Sales Prediction dataset.

The training dataset contains 8,523 rows and 12 columns, including the target column `Item_Outlet_Sales`. The test dataset contains 5,681 rows and 11 columns.

Key features include:

- `Item_Identifier`
- `Item_Weight`
- `Item_Fat_Content`
- `Item_Visibility`
- `Item_Type`
- `Item_MRP`
- `Outlet_Identifier`
- `Outlet_Size`
- `Outlet_Location_Type`
- `Outlet_Type`

The target variable is `Item_Outlet_Sales`.

**Dataset source:** [BigMart Sales Prediction — Kaggle](https://www.kaggle.com/datasets/brijbhushannanda1979/bigmart-sales-data)

Download the dataset separately and place `Train.csv` and `Test.csv` inside `data/raw/`.

## Project Structure

```text
bigmart-sales-prediction/
├── data/
│   ├── raw/
│   │   ├── Train.csv
│   │   └── Test.csv
│   └── processed/
│       ├── bigmart_sales_predictions.csv
│       └── model_comparison.csv
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   └── 04_data_preprocessing.ipynb
├── models/
├── .gitignore
└── README.md
```

## Methodology

### 1. Data Understanding
Inspected dataset dimensions, column names, data types, summary statistics, missing values, duplicates, and categorical distributions.

### 2. Data Cleaning
Standardized inconsistent item fat-content labels and handled missing numerical and categorical values.

### 3. Exploratory Data Analysis
Examined sales distributions and relationships between sales, product categories, product prices, visibility, and outlet characteristics.

### 4. Data Preprocessing
Separated features from the target, created training and validation sets, imputed missing values, scaled numerical features, and one-hot encoded categorical features.

### 5. Model Training and Evaluation
Trained and evaluated Linear Regression, Random Forest, Tuned Random Forest, and Gradient Boosting models.

### 6. Feature Engineering
Extracted product identifier prefixes and evaluated whether the additional feature improved validation performance.

### 7. Prediction Export
Used the selected model to generate predictions for all 5,681 test records and saved the results as a CSV file.

## Model Performance

The models were evaluated using the same validation split.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 944.21 | 1,273.70 | 0.4031 |
| Random Forest | 759.20 | 1,091.61 | 0.5616 |
| Tuned Random Forest | **717.54** | **1,026.54** | **0.6123** |
| Gradient Boosting | 731.31 | 1,040.98 | 0.6013 |

The Tuned Random Forest achieved the best validation performance among the four models tested. It produced the lowest MAE and RMSE and the highest R² score.

The R² score of 0.6123 indicates that the model explains approximately 61.23% of the variation in the validation target relative to a mean-prediction baseline.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook
- Git and GitHub

## How to Run the Project

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install the required packages.
4. Download the dataset and place the CSV files in `data/raw/`.
5. Open the notebooks in VS Code or Jupyter.
6. Run the notebooks in numerical order.

Example setup commands for macOS:

```bash
git clone https://github.com/YahiBae/bigmart-sales-prediction.git
cd bigmart-sales-prediction

python3 -m venv .venv
source .venv/bin/activate

python -m pip install pandas numpy matplotlib seaborn scikit-learn jupyter ipykernel
jupyter notebook
```

## Limitations

- The current best model explains approximately 61.23% of the variation in validation sales.
- Some high-sales records remain difficult to predict accurately.
- Results depend on the dataset, preprocessing decisions, and validation split.
- Further feature engineering, model tuning, and validation could improve performance.

## Future Improvements

- Investigate additional product and outlet features.
- Explore further hyperparameter tuning.
- Analyze prediction errors in greater detail.
- Develop a simple interactive interface for making predictions.

## Author

Developed as a machine learning learning project using the BigMart Sales Prediction dataset.
