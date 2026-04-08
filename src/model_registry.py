import os
import dagshub
import mlflow
import mlflow.catboost
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

dagshub.init(repo_owner='aansai', repo_name='first-mlops-end-to-end-project', mlflow=True)

os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/aansai/first-mlops-end-to-end-project.mlflow"
mlflow.set_tracking_uri("https://dagshub.com/aansai/first-mlops-end-to-end-project.mlflow")

EXPERIMENT_NAME = "Employees_classification_modelreg"
MODEL_NAME = "Employees_Catboost_model"

mlflow.set_experiment(EXPERIMENT_NAME)

client = MlflowClient()


def get_cat_features(X: pd.DataFrame):
    return [col for col in X.columns if X[col].dtype == 'object' or str(X[col].dtype) == 'category']


def load_data():
    df = pd.read_csv('C:/1ML/data/interim/df_feature.csv')

    cat_cols = get_cat_features(df)
    for col in cat_cols:
        df[col] = df[col].fillna('Unknown').astype(str)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    int_cols = df.select_dtypes(include=['int64', 'int32']).columns.tolist()
    for col in int_cols:
        df[col] = df[col].astype('float64')

    X = df.drop(columns=['EmployeeStatus'])
    y = df['EmployeeStatus']
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_model(X_train, y_train):
    cat_features = get_cat_features(X_train)
    train_pool = Pool(data=X_train, label=y_train, cat_features=cat_features)
    model = CatBoostClassifier(
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
    model.fit(train_pool)
    return model


def evaluate_model(model, X_test, y_test):
    cat_features = get_cat_features(X_test)
    test_pool = Pool(data=X_test, label=y_test, cat_features=cat_features)
    preds = model.predict(test_pool)
    return accuracy_score(y_test, preds)


def transition_model_stage(version, stage):
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version,
        stage=stage,
        archive_existing_versions=True,
    )


def add_model_description(version, description):
    client.update_model_version(
        name=MODEL_NAME,
        version=version,
        description=description,
    )


def add_registered_model_tags(tags: dict):
    for key, value in tags.items():
        client.set_registered_model_tag(MODEL_NAME, key, value)


def load_production_model():
    model_uri = f"models:/{MODEL_NAME}/Production"
    return mlflow.catboost.load_model(model_uri)


def list_model_versions():
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    for v in versions:
        print(f"  Version: {v.version} | Stage: {v.current_stage} | Run ID: {v.run_id}")


def delete_model_version(version):
    client.delete_model_version(name=MODEL_NAME, version=version)


def archive_model_version(version):
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version,
        stage="Archived",
    )


def run_pipeline():
    X_train, X_test, y_train, y_test = load_data()

    with mlflow.start_run() as run:
        model = train_model(X_train, y_train)
        accuracy = evaluate_model(model, X_test, y_test)

        mlflow.log_param("iterations", 500)
        mlflow.log_param("depth", 8)
        mlflow.log_param("learning_rate", 0.05)
        mlflow.log_metric("accuracy", accuracy)

        signature = infer_signature(X_train, model.predict(X_train))

        model_info = mlflow.catboost.log_model(
            cb_model=model,
            name="model",
            signature=signature,
            input_example=X_train.iloc[:5],
        )

        model_uri = model_info.model_uri
        run_id = run.info.run_id

    print(f"\nRun ID  : {run_id}")
    print(f"Model URI: {model_uri}")
    print(f"Accuracy: {accuracy:.4f}")

    result = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
    version = result.version
    print(f"Registered model version: {version}")

    add_model_description(
        version,
        f"CatBoost classifier for Employee classification. Accuracy: {accuracy:.4f}",
    )

    add_registered_model_tags({
        "team": "ml-team",
        "framework": "catboost",
        "task": "classification",
    })

    transition_model_stage(version, "Staging")
    print(f"Model v{version} moved to Staging")

    transition_model_stage(version, "Production")
    print(f"Model v{version} moved to Production")

    print("\nAll registered versions:")
    list_model_versions()

    prod_model = load_production_model()
    sample = X_test.iloc[:3]
    preds = prod_model.predict(sample)
    print(f"\nSample predictions from Production model: {preds}")

    return version


if __name__ == "__main__":
    version = run_pipeline()
    print(f"\nPipeline complete. Latest version: {version}")