# PRODIGY_DS_03
# Bank Marketing Dataset Analysis and Decision Tree Classifier

This project performs data analysis and builds a **Decision Tree Classifier** using the **UCI Bank Marketing Dataset**. The dataset contains information about marketing campaigns of a Portuguese banking institution, and the goal is to predict whether a client will subscribe to a term deposit.

## Project Overview

1. **Data Preprocessing**: Handling missing values and encoding categorical features.
2. **Model Building**: Implementing a Decision Tree classifier to predict term deposit subscription.
3. **Model Evaluation**: Evaluating the performance of the classifier using accuracy, confusion matrix, classification report, and ROC AUC score.

## Dataset

The dataset is downloaded from the UCI Machine Learning Repository:
- URL: [https://archive.ics.uci.edu/ml/datasets/bank+marketing](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- The dataset is stored in a ZIP file and contains two CSV files: `bank-additional-full.csv` (used in this project) and `bank-additional.csv`.

### Key Features:

- `age`: Age of the client
- `job`: Type of job
- `marital`: Marital status
- `education`: Level of education
- `default`: Has credit in default?
- `housing`: Has housing loan?
- `loan`: Has personal loan?
- `contact`: Contact communication type
- `month`: Last contact month of year
- `day_of_week`: Last contact day of the week
- `duration`: Last contact duration, in seconds
- `campaign`: Number of contacts performed during this campaign
- `pdays`: Number of days since the client was last contacted
- `previous`: Number of contacts performed before this campaign
- `poutcome`: Outcome of the previous marketing campaign
- `emp.var.rate`: Employment variation rate
- `cons.price.idx`: Consumer price index
- `cons.conf.idx`: Consumer confidence index
- `euribor3m`: Euribor 3 month rate
- `nr.employed`: Number of employees
- `y`: Whether the client subscribed to a term deposit (target variable)

## Requirements

Ensure the following Python packages are installed:

- `pandas`
- `scikit-learn`

Install them using:

```bash
pip install pandas scikit-learn
