import logging
import numpy as np
import pandas as pd
import os
from datetime import datetime
from src.logger import logger 

logger.debug("Logger initialized and configuration loaded.")
logger.info("Starting the Data Preprocessing Pipeline...")
logger.warning("External data source detected: checking file integrity.")
logger.error("Data validation failed: missing columns in input CSV.")
logger.critical("Memory limit reached: process terminated.")

def data_load(url):
    df = pd.read_csv(url)
    logger.info("Data load Successfully")
    return df

def datetime_fix(df):
    try:
        df['DOB'] = pd.to_datetime(df['DOB'], errors='coerce')
        df['Survey Date'] = pd.to_datetime(df['Survey Date'], errors='coerce')
        return df
    except Exception as e:
        logger.exception("Error in datetime_fix")
        raise

def start_date(df):
    try:
        df['StartDate'] = pd.to_datetime(df['StartDate'], errors='coerce')
        return df
    except Exception as e:
        logger.exception("Error in start_date")
        raise

def training_date(df):
    try:
        df['Training Date'] = pd.to_datetime(df['Training Date'], errors='coerce')
        return df
    except Exception as e:
        logger.exception("Error in training_date")
        raise

def tenure_day(df):
    try:
        reference_date = pd.Timestamp.today()
        df["tenure_days"] = (reference_date - df["StartDate"]).dt.days
        df["tenure_years"] = df["tenure_days"] / 365.25
        return df
    except Exception as e:
        logger.exception("Error in tenure_day")
        raise

def age(df):
    try:
        reference_date = pd.Timestamp.today()
        df["age"] = (reference_date - df["DOB"]).dt.days / 365.25
        df["age"] = df["age"].clip(lower=18, upper=80)
        return df
    except Exception as e:
        logger.exception("Error in age")
        raise

def composite_wellness(df):
    try:
        df["composite_wellness_score"] = (
            df["Engagement Score"] + 
            df["Satisfaction Score"] +
            df["Work-Life Balance Score"]
        ) / 3
        return df
    except Exception as e:
        logger.exception("Error in composite_wellness")
        raise  

def days_since(df):
    try:
        reference_date = pd.Timestamp.today()
        df["days_since_training"] = (reference_date - df["Training Date"]).dt.days
        df["days_since_training"] = df["days_since_training"].clip(lower=0)
        return df
    except Exception as e:
        logger.exception("Error in days_since")
        raise

def training_cost(df):
    try:
        df["training_cost_per_day"] = df["Training Cost"] / (
            df["Training Duration(Days)"].replace(0, np.nan)
        )
        df["training_cost_per_day"] = df["training_cost_per_day"].fillna(0)
        return df
    except Exception as e:
        logger.exception("Error in training_cost")
        raise

def is_early_tenure(df):
    try:
        df["is_early_tenure"] = (df["tenure_days"] <= 365).astype(int)
        return df
    except Exception as e:
        logger.exception("Error in is_early_tenure")
        raise

def survey_lag(df):
    try:
        df["survey_lag_days"] = (df["Survey Date"] - df["StartDate"]).dt.days
        df["survey_lag_days"] = df["survey_lag_days"].clip(lower=0)
        return df
    except Exception as e:
        logger.exception("Error in survey_lag")
        raise

def is_disengaged(df):
    try:
        df["is_disengaged"] = (df["Engagement Score"] <= 2).astype(int)
        df["is_dissatisfied"] = (df["Satisfaction Score"] <= 2).astype(int)
        df["is_wlb_poor"] = (df["Work-Life Balance Score"] <= 2).astype(int)
        return df
    except Exception as e:
        logger.exception("Error in is_disengaged")
        raise

def any_low_sentiment(df):
    try:
        df["any_low_sentiment"] = (
            (df["is_disengaged"] + df["is_dissatisfied"] + df["is_wlb_poor"]) >= 2
        ).astype(int)
        return df
    except Exception as e:
        logger.exception("Error in any_low_sentiment")
        raise

def month(df):
    try:
        df["start_month"] = df["StartDate"].dt.month
        df["start_quarter"] = df["StartDate"].dt.quarter
        return df
    except Exception as e:
        logger.exception("Error in month")
        raise

def age_is_hire(df):
    try:
        df["age_at_hire"] = (df["StartDate"] - df["DOB"]).dt.days / 365.25
        df["age_at_hire"] = df["age_at_hire"].clip(lower=16, upper=75)
        return df
    except Exception as e:
        logger.exception("Error in age_is_hire")
        raise

def drop_cols(df):
    try:
        df.drop(['DOB', 'age', 'age_at_hire', 'LocationCode'], axis=1, inplace=True, errors='ignore')
        return df
    except Exception as e:
        logger.exception("Error in drop_cols")
        raise

def quantile(df):
    try:
        q1 = df['training_cost_per_day'].quantile(0.25)
        q3 = df['training_cost_per_day'].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        higher = q3 + 1.5 * iqr
        df['training_cost_per_day'] = df['training_cost_per_day'].clip(lower, higher)
        return df
    except Exception as e:
        logger.exception("Error in quantile")
        raise

def save_data(data_path, df):
    try:
        save_path = os.path.join(data_path, "interim")
        os.makedirs(save_path, exist_ok=True)
        df.to_csv(os.path.join(save_path, "df_feature.csv"), index=False)
        logger.info("Data saved successfully")
    except Exception as e:
        logger.exception(f"Error in save_data: {str(e)}")
        raise


def main():
    try:
        df = data_load(r"C:\1ML\data\interim\df_clean.csv")

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
        df = age_is_hire(df)
        df = drop_cols(df)
        df = quantile(df)

        save_data(r"C:\1ML\data", df)

    except Exception as e:
        logger.exception("Error in main")
        raise


if __name__ == "__main__":
    main()