import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from src.logger import logger
import os
import mlflow
import yaml
import joblib
import dagshub

# ================= DAGSHUB + MLFLOW =================
dagshub.init(repo_owner='aansai', repo_name='first-mlops-end-to-end-project', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/aansai/first-mlops-end-to-end-project.mlflow')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
PARAMS_PATH = os.path.join(ROOT_DIR, "params.yaml")

with open(PARAMS_PATH, "r") as file:
    params = yaml.safe_load(file)

model_params = params["model"]

logger.info("Starting the Data Preprocessing Pipeline...")

# ================= LOAD DATA =================
def load_data(url):
    logger.info('Data Loading Start')
    df = pd.read_csv(url)
    logger.info('Data Load Successfully')
    return df

# ================= FIXED TARGET =================
def prepare_target(df):
    logger.info("Preparing target column")

    print("Unique values BEFORE processing:", df['EmployeeStatus'].unique())

    # Convert to numeric if already 0/1
    df['EmployeeStatus'] = pd.to_numeric(df['EmployeeStatus'], errors='coerce')

    print("Unique values AFTER conversion:", df['EmployeeStatus'].unique())

    df = df.dropna(subset=['EmployeeStatus'])

    if df.empty:
        raise ValueError("❌ Dataset became empty after target preparation")

    df['EmployeeStatus'] = df['EmployeeStatus'].astype(int)

    return df

# ================= SPLIT =================
def data_split(df):
    logger.info('Data Splitting Start')

    X = df.drop(columns=['EmployeeStatus'])
    y = df['EmployeeStatus']

    if df.empty:
        raise ValueError("❌ Dataset is empty before splitting")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    logger.info('Data Splitting Done')
    return X_train, X_test, y_train, y_test

# ================= FEATURES =================
num_cols = [
    'Current Employee Rating',
    'Engagement Score',
    'Satisfaction Score',
    'Work-Life Balance Score',
    'Training Duration(Days)',
    'Training Cost',
    'tenure_days',
    'tenure_years',
    'composite_wellness_score',
    'days_since_training',
    'training_cost_per_day',
    'survey_lag_days'
]

cat_cols = [
    'BusinessUnit',
    'EmployeeType',
    'PayZone',
    'EmployeeClassificationType',
    'DepartmentType',
    'State',
    'GenderCode',
    'RaceDesc',
    'MaritalDesc',
    'Training Type',
    'Training Outcome',
    'Title_Category',
    'Division_Category',
    'JobFunction_Category'
]

# ================= PREPROCESSING =================
def processing(num_cols, cat_cols):
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    processor = ColumnTransformer([
        ('cat', cat_pipeline, cat_cols),
        ('num', num_pipeline, num_cols)
    ])

    return processor

# ================= MODEL =================
def model_build(X_train, X_test, y_train, y_test, processor):
    logger.info('Building a Model Start')

    model = CatBoostClassifier(**model_params)

    pipe = ImbPipeline([
        ('preprocessor', processor),
        ('smote', SMOTE(random_state=42)),
        ('model', model)
    ])

    mlflow.set_experiment('employees-status-classification-v2')
    mlflow.autolog()

    with mlflow.start_run():
        pipe.fit(X_train, y_train)

        y_pred_prob = pipe.predict_proba(X_test)
        y_pred = (y_pred_prob[:, 1] > 0.5).astype(int)

        report = classification_report(y_test, y_pred, output_dict=True)

        mlflow.log_metric('accuracy', report['accuracy'])
        mlflow.log_metric('precision', report['weighted avg']['precision'])
        mlflow.log_metric('recall', report['weighted avg']['recall'])
        mlflow.log_metric('f1-score', report['weighted avg']['f1-score'])

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()

        plt.savefig('confusion_matrix.png')
        mlflow.log_artifact('confusion_matrix.png')

        mlflow.set_tags({
            'Author': 'Anas',
            'Project': 'employees-status-classification'
        })

        print(classification_report(y_test, y_pred))

    logger.info('Building A Model Successfully')
    return pipe

# ================= SAVE MODEL =================
def save_data(model):
    save_path = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "..", "data", "model"))
    os.makedirs(save_path, exist_ok=True)

    joblib.dump(model, os.path.join(save_path, 'model.pkl'))
    logger.info('Model Saved Successfully')

# ================= MAIN =================
def main():
    df = load_data('C:/1ML/data/interim/df_feature.csv')

    print("Initial shape:", df.shape)
    print("EmployeeStatus unique BEFORE processing:", df['EmployeeStatus'].unique())

    df = prepare_target(df)

    X_train, X_test, y_train, y_test = data_split(df)

    print("Target unique values:", df['EmployeeStatus'].unique())

    processor = processing(num_cols, cat_cols)

    model = model_build(X_train, X_test, y_train, y_test, processor)

    save_data(model)

# ================= RUN =================
if __name__ == '__main__':
    main()