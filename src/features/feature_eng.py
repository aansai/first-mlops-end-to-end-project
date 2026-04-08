import numpy as np
import pandas as pd
import os
import yaml
# from src.connections.s3_connection import S3Connection  # COMMENTED
from src.logger import logger

def load_config():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def data_load(url):
    try:
        df = pd.read_csv(url)
        logger.info(f"Data loaded successfully — shape: {df.shape}")
        return df
    except Exception as e:
        logger.exception(f"Error in data_load: {str(e)}")
        raise

def datetime_fix(df):
    try:
        if 'DOB' in df.columns:
            df['DOB'] = pd.to_datetime(df['DOB'], errors='coerce')
        else:
            logger.warning("DOB column missing")
        if 'Survey Date' in df.columns:
            df['Survey Date'] = pd.to_datetime(df['Survey Date'], errors='coerce')
        else:
            logger.warning("Survey Date column missing")
        return df
    except Exception as e:
        logger.exception(f"Error in datetime_fix: {str(e)}")
        raise

def start_date(df):
    try:
        if 'StartDate' in df.columns:
            df['StartDate'] = pd.to_datetime(df['StartDate'], errors='coerce')
        else:
            logger.warning("StartDate missing")
        return df
    except Exception as e:
        logger.exception(f"Error in start_date: {str(e)}")
        raise

def training_date(df):
    try:
        if 'Training Date' in df.columns:
            df['Training Date'] = pd.to_datetime(df['Training Date'], errors='coerce')
        else:
            logger.warning("Training Date missing")
        return df
    except Exception as e:
        logger.exception(f"Error in training_date: {str(e)}")
        raise

def tenure_day(df):
    try:
        if 'StartDate' in df.columns:
            ref = pd.Timestamp.today()
            df["tenure_days"] = (ref - df["StartDate"]).dt.days
            df["tenure_years"] = df["tenure_days"] / 365.25
        return df
    except Exception as e:
        logger.exception(f"Error in tenure_day: {str(e)}")
        raise

def age(df):
    try:
        if 'DOB' in df.columns:
            ref = pd.Timestamp.today()
            df["age"] = (ref - df["DOB"]).dt.days / 365.25
            df["age"] = df["age"].clip(18, 80)
        else:
            logger.warning("DOB missing, skipping age")
        return df
    except Exception as e:
        logger.exception(f"Error in age: {str(e)}")
        raise

def composite_wellness(df):
    try:
        cols = ['Engagement Score', 'Satisfaction Score', 'Work-Life Balance Score']
        if all(col in df.columns for col in cols):
            df["composite_wellness_score"] = df[cols].mean(axis=1)
        else:
            logger.warning("Wellness score columns missing")
        return df
    except Exception as e:
        logger.exception(f"Error in composite_wellness: {str(e)}")
        raise

def days_since(df):
    try:
        if 'Training Date' in df.columns:
            ref = pd.Timestamp.today()
            df["days_since_training"] = (ref - df["Training Date"]).dt.days
            df["days_since_training"] = df["days_since_training"].clip(lower=0)
        return df
    except Exception as e:
        logger.exception(f"Error in days_since: {str(e)}")
        raise

def training_cost(df):
    try:
        if 'Training Cost' in df.columns and 'Training Duration(Days)' in df.columns:
            df["training_cost_per_day"] = df["Training Cost"] / (
                df["Training Duration(Days)"].replace(0, np.nan)
            )
            df["training_cost_per_day"] = df["training_cost_per_day"].fillna(0)
        return df
    except Exception as e:
        logger.exception(f"Error in training_cost: {str(e)}")
        raise

def is_early_tenure(df):
    try:
        if 'tenure_days' in df.columns:
            df["is_early_tenure"] = (df["tenure_days"] <= 365).astype(int)
        return df
    except Exception as e:
        logger.exception(f"Error in is_early_tenure: {str(e)}")
        raise

def survey_lag(df):
    try:
        if 'Survey Date' in df.columns and 'StartDate' in df.columns:
            df["survey_lag_days"] = (df["Survey Date"] - df["StartDate"]).dt.days
            df["survey_lag_days"] = df["survey_lag_days"].clip(lower=0)
        return df
    except Exception as e:
        logger.exception(f"Error in survey_lag: {str(e)}")
        raise

def is_disengaged(df):
    try:
        if 'Engagement Score' in df.columns:
            df["is_disengaged"] = (df["Engagement Score"] <= 2).astype(int)
        if 'Satisfaction Score' in df.columns:
            df["is_dissatisfied"] = (df["Satisfaction Score"] <= 2).astype(int)
        if 'Work-Life Balance Score' in df.columns:
            df["is_wlb_poor"] = (df["Work-Life Balance Score"] <= 2).astype(int)
        return df
    except Exception as e:
        logger.exception(f"Error in is_disengaged: {str(e)}")
        raise

def any_low_sentiment(df):
    try:
        if all(col in df.columns for col in ["is_disengaged", "is_dissatisfied", "is_wlb_poor"]):
            df["any_low_sentiment"] = (
                (df["is_disengaged"] + df["is_dissatisfied"] + df["is_wlb_poor"]) >= 2
            ).astype(int)
        return df
    except Exception as e:
        logger.exception(f"Error in any_low_sentiment: {str(e)}")
        raise

def month(df):
    try:
        if 'StartDate' in df.columns:
            df["start_month"] = df["StartDate"].dt.month
            df["start_quarter"] = df["StartDate"].dt.quarter
        return df
    except Exception as e:
        logger.exception(f"Error in month: {str(e)}")
        raise

def age_at_hire(df):
    try:
        if 'DOB' in df.columns and 'StartDate' in df.columns:
            df["age_at_hire"] = (df["StartDate"] - df["DOB"]).dt.days / 365.25
            df["age_at_hire"] = df["age_at_hire"].clip(16, 75)
        return df
    except Exception as e:
        logger.exception(f"Error in age_at_hire: {str(e)}")
        raise

def employee_status(df):
    try:
        if 'EmployeeStatus' in df.columns:
            df = df[df['EmployeeStatus'] != 'Future Start'].copy()
            attrition_map = {
                'Active': 0,
                'Leave of Absence': 0,
                'Voluntarily Terminated': 1,
                'Terminated for Cause': 1
            }
            df['EmployeeStatus'] = df['EmployeeStatus'].map(attrition_map)
            if df['EmployeeStatus'].isna().sum() > 0:
                logger.warning("Unmapped EmployeeStatus values found")
        return df
    except Exception as e:
        logger.exception(f"Error in employee_status: {str(e)}")
        raise

def drop_cols(df):
    try:
        df.drop(['DOB', 'age', 'age_at_hire', 'LocationCode'], axis=1, inplace=True, errors='ignore')
        return df
    except Exception as e:
        logger.exception(f"Error in drop_cols: {str(e)}")
        raise

def quantile(df):
    try:
        if 'training_cost_per_day' in df.columns:
            q1 = df['training_cost_per_day'].quantile(0.25)
            q3 = df['training_cost_per_day'].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            higher = q3 + 1.5 * iqr
            df['training_cost_per_day'] = df['training_cost_per_day'].clip(lower, higher)
        return df
    except Exception as e:
        logger.exception(f"Error in quantile: {str(e)}")
        raise

def save_data(data_path, df):
    try:
        save_path = os.path.join(data_path, "featured")
        os.makedirs(save_path, exist_ok=True)
        output_file = os.path.join(save_path, "df_feature.csv")
        df.to_csv(output_file, index=False)
        logger.info(f"Data saved locally at: {output_file}")
        return output_file
    except Exception as e:
        logger.exception(f"Error in save_data: {str(e)}")
        raise

# def download_from_s3(bucket_name, s3_key, local_path):
#     try:
#         os.makedirs(os.path.dirname(local_path), exist_ok=True)
#         s3 = S3Connection()
#         s3.download_file(bucket_name, s3_key, local_path)
#         logger.info(f"Downloaded from s3://{bucket_name}/{s3_key}")
#     except Exception as e:
#         logger.exception(f"Error in download_from_s3: {str(e)}")
#         raise

# def upload_to_s3(local_file, bucket_name, s3_key):
#     try:
#         s3 = S3Connection()
#         s3.upload_file(local_file, bucket_name, s3_key)
#         logger.info(f"Uploaded to s3://{bucket_name}/{s3_key}")
#     except Exception as e:
#         logger.exception(f"Error in upload_to_s3: {str(e)}")
#         raise

def main():
    try:
        logger.info("Starting Feature Engineering Pipeline...")
        config = load_config()["feature_engineering"]

        # local_input = "data/interim/df_clean.csv"
        # download_from_s3(
        #     bucket_name=config["bucket_name"],
        #     s3_key=config["input_s3_key"],
        #     local_path=local_input
        # )

        local_input = "data/interim/df_clean.csv"  # use local file

        df = data_load(local_input)

        logger.info(f"Columns: {df.columns.tolist()}")

        df = datetime_fix(df)
        df = start_date(df)
        df = training_date(df)
        df = tenure_day(df)
        df = age(df)
        df = composite_wellness(df)
        df = days_since(df)
        df = training_cost(df)
        df = is_early_tenure(df)
        df = survey_lag(df)
        df = is_disengaged(df)
        df = any_low_sentiment(df)
        df = month(df)
        df = age_at_hire(df)
        df = employee_status(df)
        df = drop_cols(df)
        df = quantile(df)

        local_file = save_data(config["local_save_path"], df)

        # upload_to_s3(
        #     local_file=local_file,
        #     bucket_name=config["bucket_name"],
        #     s3_key=config["output_s3_key"]
        # )

        logger.info("Feature Engineering Pipeline completed successfully.")

    except Exception as e:
        logger.critical(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()