
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer



df = pd.read_csv("employee_attrition_raw.csv")

print("Original dataset shape:", df.shape)


columns_to_drop = [
    'EmployeeCount',
    'EmployeeNumber',
    'Over18',
    'StandardHours'
]

df_clean = df.drop(columns=columns_to_drop)


df_clean['CompanyExperienceRatio'] = (
    df_clean['YearsAtCompany'] /
    df_clean['TotalWorkingYears'].replace(0, np.nan)
)

df_clean['CompanyExperienceRatio'] = (
    df_clean['CompanyExperienceRatio'].fillna(0)
)

df_clean['CareerStage'] = pd.cut(
    df_clean['TotalWorkingYears'],
    bins=[-1, 5, 10, 20, np.inf],
    labels=[
        'Early Career',
        'Developing',
        'Experienced',
        'Senior'
    ]
)



X = df_clean.drop('Attrition', axis=1)
y = df_clean['Attrition']


numeric_features = X.select_dtypes(
    include=['int64', 'float64']
).columns.tolist()

categorical_features = X.select_dtypes(
    include=['object', 'category']
).columns.tolist()



numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(
        handle_unknown='ignore',
        sparse_output=False
    ))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)




X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)



df_clean.to_csv(
    "employee_attrition_clean.csv",
    index=False
)

joblib.dump(
    preprocessor,
    "preprocessor.pkl"
)


print("\nPreprocessing completed successfully!")
print("Clean dataset shape:", df_clean.shape)
print("Training data shape:", X_train_processed.shape)
print("Testing data shape:", X_test_processed.shape)
print("Saved: employee_attrition_clean.csv")
print("Saved: preprocessor.pkl")
