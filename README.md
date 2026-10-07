AI-ML Task 2 – Data Preprocessing, Feature Engineering & Machine Learning Pipeline

Overview
This project was completed as part of my AI/ML internship Task 2. The main aim of this task was to preprocess the IBM HR Analytics Employee Attrition dataset and prepare it for machine learning.
The project covers data cleaning, feature engineering, outlier detection, categorical encoding, feature scaling, train-test splitting, and development of a reusable preprocessing pipeline.

Dataset
The dataset used for this task is the IBM HR Analytics Employee Attrition Dataset.
- Number of records: 1470
- Original number of columns: 35
- Target variable: `Attrition`
The target variable indicates whether an employee has left the organization (`Yes` or `No`).

Work Completed

1. Dataset Understanding
- Loaded and inspected the dataset.
- Checked the number of rows and columns.
- Examined data types.
- Checked for missing values.
- Checked for duplicate records.
- Examined categorical values and numerical features.

2. Data Cleaning
The following unnecessary columns were removed:
- `EmployeeCount`
- `EmployeeNumber`
- `Over18`
- `StandardHours`
Missing values and duplicate records were checked during preprocessing.

3. Feature Engineering
Two new features were created:
- CompanyExperienceRatio – represents the proportion of total working experience spent at the current company.
- CareerStage– groups employees based on their total working years into:
  - Early Career
  - Developing
  - Experienced
  - Senior

4. Outlier Detection
Outliers were checked using the IQR method and box plots for selected numerical features such as:
- Age
- Monthly Income
- Total Working Years
- Years at Company

The detected values were reviewed rather than automatically removed because some extreme values can still represent valid employee records.

5. Encoding and Scaling
Categorical features were converted into numerical form using One-Hot Encoding.
Numerical features were scaled using StandardScaler.
A comparison between StandardScaler and MinMaxScaler was also performed as part of the task exercises.

6. Train-Test Split
The dataset was divided into training and testing sets using:
- Test size: 20%
- Random state: 42
- Stratification based on the target variable

The resulting datasets were:
- Training data: 1176 records
- Testing data: 294 records

7. Machine Learning Preprocessing Pipeline
A reusable preprocessing pipeline was developed using:

- Pipeline
- ColumnTransformer
- SimpleImputer
- OneHotEncoder
- StandardScaler
The pipeline handles numerical and categorical features separately and can also handle previously unseen categorical values.

8. Saving the Pipeline
The completed preprocessing pipeline was saved using Joblib as:
preprocessor.pkl
The preprocessing code was also saved as:
preprocessing_pipeline.py

Repository Contents

File  Description

02_data_preprocessing.ipynb Complete preprocessing notebook
preprocessing_pipeline.py Reusable preprocessing script
employee_attrition_clean.csv Cleaned dataset with engineered features 
preprocessor.pkl  Saved preprocessing pipeline 
Preprocessing_Report.pdf Report describing the preprocessing work 

Tools and Libraries Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Google Colab
- GitHub

Final Output
The preprocessing workflow converts the original employee attrition dataset into a machine-learning-ready format.
The saved preprocessing pipeline can be reused to apply the same preprocessing steps to new data.

 Learning Outcomes
Through this task, I practiced:
- Data cleaning and validation
- Feature engineering
- Categorical encoding
- Feature scaling
- Outlier detection
- Train-test splitting
- Building reusable preprocessing pipelines
- Saving and reloading preprocessing models
- Organizing an ML preprocessing project using GitHub
