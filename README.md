# Waze Churn Prediction Notebook README

## Overview
This notebook focuses on analyzing user data from the Waze app to predict user churn. The goal is to identify factors contributing to churn and build a machine learning model that can accurately classify users as 'retained' or 'churned'.

## Data Analysis Steps

1.  **Data Loading and Initial Inspection**: The `waze_dataset.csv` file was loaded into a pandas DataFrame. Initial checks were performed using `df.head()`, `df.info()`, and `df.describe()` to understand the data structure, types, and summary statistics.

2.  **Feature Engineering**: Several new features were created to enhance the predictive power of the model:
    *   `km_per_driving_day`: Kilometers driven per driving day.
    *   `session_percentage`: Percentage of sessions out of total sessions.
    *   `professional_driver`: A binary indicator (1 if a driver has more than 50 drives and drove more than 15 days, 0 otherwise).
    *   `total_sessions_per_day`: Total sessions per day after onboarding.
    *   `km_per_hour`: Kilometers driven per hour.
    *   `km_per_driver`: Kilometers driven per drive.
    *   `percent_of_fav`: Percentage of total navigations to favorite locations out of total sessions.

3.  **Data Cleaning**: Missing values in the 'label' column were dropped. Infinite values in engineered features were replaced with 0 to ensure numerical stability.

4.  **Data Preprocessing**: 
    *   The 'device' column (categorical) was encoded into numerical format (Android=1, iPhone=0).
    *   The 'label' column (target variable) was encoded (churned=1, retained=0).
    *   The 'ID' column was dropped as it's not relevant for modeling.

5.  **Data Splitting**: The dataset was split into training, validation, and testing sets using `train_test_split` with stratification to maintain the class distribution of the target variable (`label`).

## Model Training and Evaluation

Two machine learning models were trained and evaluated:

### 1. Random Forest Classifier
*   **Hyperparameter Tuning**: `RandomizedSearchCV` was used to find the best hyperparameters for a `RandomForestClassifier`.
*   **Evaluation Metrics**: The model was evaluated based on accuracy, precision, recall, and F1-score on both the validation and test sets.

### 2. XGBoost Classifier
*   **Hyperparameter Tuning**: `RandomizedSearchCV` was used to find the best hyperparameters for an `XGBClassifier`.
*   **Evaluation Metrics**: Similar to Random Forest, the model was evaluated on accuracy, precision, recall, and F1-score on both the validation and test sets.

## Results
The performance of both models on the validation and test sets is summarized in the `results` DataFrame.

**Best Performing Model**: Based on the recall score, the XGBoost Classifier performed slightly better on the cross-validation set. The final evaluation on the test set confirmed its performance.

**Confusion Matrix**: A confusion matrix was generated for the XGBoost model on the test set to visualize its classification performance, showing true positives, true negatives, false positives, and false negatives.

**Feature Importance**: The feature importance plot from the best XGBoost estimator highlighted the most influential features in predicting user churn, providing insights into which factors are most critical.

### Summary of Scores:

| model      | precision | recall    | F1        | accuracy  |
|:-----------|:----------|:----------|:----------|:----------|
| RF cv      | 0.519939  | 0.104142  | 0.173326  | 0.823954  |
| XGBoost cv | 0.443440  | 0.165751  | 0.241130  | 0.815211  |
| RF val     | 0.433333  | 0.096059  | 0.157258  | 0.817308  |
| XGBoost val| 0.400000  | 0.142857  | 0.210526  | 0.809878  |
| XGB test   | 0.407767  | 0.165680  | 0.235624  | 0.809441  |

