import mlflow
import mlflow.sklearn
import dagshub

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

dagshub.init(repo_owner='aansai', repo_name='first-mlops-end-to-end-project', mlflow=True)

mlflow.set_tracking_uri('https://dagshub.com/aansai/first-mlops-end-to-end-project.mlflow')
mlflow.set_experiment("employee_status_classification")


def evaluate_model(name, model, X_train, X_test, y_train, y_test, processor):

    pipe = ImbPipeline([
        ('preprocessor', processor),
        ('smote', SMOTE(random_state=42)),
        ('model', model)
    ])

    with mlflow.start_run(run_name=name):

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

    if hasattr(pipe, "predict_proba"):
        y_prob = pipe.predict_proba(X_test)

        # Ensure it's 2D
        if len(y_prob.shape) == 2:
            auc = roc_auc_score(
                y_test,
                y_prob,
                multi_class='ovr',
                average='macro'
            )
        else:
            auc = 0  # fallback
    else:
        auc = 0


        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_param("model", name)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)

        mlflow.sklearn.log_model(pipe, name)

        return acc, prec, rec, f1, auc


def run_experiments(models, X_train, X_test, y_train, y_test, processor):

    results = []

    for name, model in models:
        metrics = evaluate_model(name, model, X_train, X_test, y_train, y_test, processor)
        results.append((name, metrics))

    return results


rf = RandomForestClassifier(
    n_estimators=500, max_depth=12, min_samples_split=5,
    min_samples_leaf=2, max_features='sqrt',
    class_weight='balanced', bootstrap=True, oob_score=True,
    random_state=42, n_jobs=-1
)

xgb = XGBClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    min_child_weight=3, gamma=0.1,
    reg_alpha=0.1, reg_lambda=1.5,
    eval_metric='logloss',
    random_state=42, n_jobs=-1
)

lgbm = LGBMClassifier(
    n_estimators=500, max_depth=8, learning_rate=0.05,
    num_leaves=63, subsample=0.8, colsample_bytree=0.8,
    min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
    class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1
)

cat = CatBoostClassifier(
    iterations=500,
    depth=8,
    learning_rate=0.05,
    l2_leaf_reg=3,
    bagging_temperature=0.8,
    random_strength=1,
    border_count=128,
    auto_class_weights='Balanced',

    # ✅ IMPORTANT: multiclass
    loss_function='MultiClass',
    eval_metric='MultiClass',

    random_seed=42,
    verbose=0
)

et = ExtraTreesClassifier(
    n_estimators=500, max_depth=12, min_samples_split=5,
    min_samples_leaf=2, max_features='sqrt',
    class_weight='balanced', random_state=42, n_jobs=-1
)

gbm = GradientBoostingClassifier(
    n_estimators=300, learning_rate=0.05,
    subsample=0.8, min_samples_split=5,
    min_samples_leaf=2, random_state=42
)

lr = LogisticRegression(
    C=0.1, solver='lbfgs', max_iter=1000,
    class_weight='balanced', random_state=42
)

models = [
    ('rf', rf),
    ('xgb', xgb),
    ('lgbm', lgbm),
    ('cat', cat),
    ('gbm', gbm),
    ('et', et),
    ('lr', lr),
    ('ada', AdaBoostClassifier(n_estimators=200, learning_rate=0.05, random_state=42))
]


if __name__ == "__main__":
    import pandas as pd
    from sklearn.model_selection import train_test_split

    from src.model.model_eval import processing

    df = pd.read_csv('C:/1ML/data/interim/df_feature.csv')

    # Features and target
    X = df.drop(columns=['EmployeeStatus'])

    # ✅ Encode target
    le = LabelEncoder()
    y = le.fit_transform(df['EmployeeStatus'])

    # (Optional but recommended) save encoder
    import joblib
    joblib.dump(le, "label_encoder.pkl")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    processor = processing(
        num_cols=[
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
        ],
        cat_cols=[
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
    )

    run_experiments(models, X_train, X_test, y_train, y_test, processor)