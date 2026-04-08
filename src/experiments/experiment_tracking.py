import mlflow
import mlflow.sklearn
import dagshub
import pandas as pd
import joblib

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from src.model.model_eval import processing

# -------------------------
# DAGSHUB + MLFLOW SETUP
# -------------------------
dagshub.init(
    repo_owner='aansai',
    repo_name='first-mlops-end-to-end-project',
    mlflow=True
)

mlflow.set_tracking_uri(
    'https://dagshub.com/aansai/first-mlops-end-to-end-project.mlflow'
)

mlflow.set_experiment("employes_status_classification")

print("Tracking URI:", mlflow.get_tracking_uri())
print("Experiment:", mlflow.get_experiment_by_name("employes_status_classification"))



def evaluate_model(name, model, X_train, X_test, y_train, y_test, processor):

    pipe = ImbPipeline([
        ('preprocessor', processor),
        ('smote', SMOTE(random_state=42)),
        ('model', model)
    ])

    acc, prec, rec, f1, auc = 0, 0, 0, 0, 0

    with mlflow.start_run(run_name=name):
        try:
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            auc = 0
            if hasattr(pipe, "predict_proba"):
                try:
                    y_prob = pipe.predict_proba(X_test)
                    if len(y_prob.shape) == 2 and y_prob.shape[1] > 1:
                        auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
                except Exception as auc_err:
                    print(f"[{name}] AUC calculation failed: {auc_err}")
                    auc = 0

            acc  = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
            rec  = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1   = f1_score(y_test, y_pred, average='macro', zero_division=0)

            # -------- LOG PARAMS --------
            mlflow.log_param("model_name", name)

            try:
                raw_params = pipe.named_steps['model'].get_params()
                # Convert every value to string to avoid serialization errors
                safe_params = {k: str(v) for k, v in raw_params.items()}
                mlflow.log_params(safe_params)
            except Exception as param_err:
                print(f"[{name}] Could not log params: {param_err}")

            # -------- LOG METRICS --------
            mlflow.log_metric("accuracy",  acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall",    rec)
            mlflow.log_metric("f1_score",  f1)
            mlflow.log_metric("roc_auc",   auc)

            # -------- LOG MODEL --------
            mlflow.sklearn.log_model(pipe, "model")

            print(f"[{name}] ✓ acc={acc:.4f} | prec={prec:.4f} | rec={rec:.4f} | f1={f1:.4f} | auc={auc:.4f}")

        except Exception as e:
            print(f"[{name}] ✗ Run FAILED with error: {e}")
            mlflow.set_tag("run_status", "FAILED")
            mlflow.set_tag("error_message", str(e))
            raise

    return acc, prec, rec, f1, auc


# -------------------------
# RUN ALL EXPERIMENTS
# -------------------------
def run_experiments(models, X_train, X_test, y_train, y_test, processor):

    results = []

    for name, model in models:
        print(f"\n{'='*50}")
        print(f"Running: {name}")
        print(f"{'='*50}")
        try:
            metrics = evaluate_model(name, model, X_train, X_test, y_train, y_test, processor)
            results.append((name, metrics))
        except Exception as e:
            print(f"[{name}] Skipped due to error: {e}")
            results.append((name, (0, 0, 0, 0, 0)))

    # -------- SUMMARY TABLE --------
    print("\n\n" + "="*70)
    print(f"{'Model':<10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}")
    print("="*70)
    for name, (acc, prec, rec, f1, auc) in results:
        print(f"{name:<10} {acc:>10.4f} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f} {auc:>10.4f}")
    print("="*70)

    return results


# -------------------------
# DEFINE MODELS
# -------------------------
rf = RandomForestClassifier(
    n_estimators=500, max_depth=12, min_samples_split=5,
    min_samples_leaf=2, max_features='sqrt',
    class_weight='balanced', random_state=42, n_jobs=-1
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

ada = AdaBoostClassifier(
    n_estimators=200, learning_rate=0.05, random_state=42
)

models = [
    ('rf',   rf),
    ('xgb',  xgb),
    ('lgbm', lgbm),
    ('cat',  cat),
    ('gbm',  gbm),
    ('et',   et),
    ('lr',   lr),
    ('ada',  ada),
]


# -------------------------
# MAIN EXECUTION
# -------------------------
if __name__ == "__main__":

    df = pd.read_csv('C:/1ML/data/interim/df_feature.csv')

    X = df.drop(columns=['EmployeeStatus'])

    le = LabelEncoder()
    y = le.fit_transform(df['EmployeeStatus'])

    joblib.dump(le, "label_encoder.pkl")
    print("LabelEncoder saved to label_encoder.pkl")
    print(f"Classes: {le.classes_}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    print(f"\nTrain size: {X_train.shape}, Test size: {X_test.shape}")

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