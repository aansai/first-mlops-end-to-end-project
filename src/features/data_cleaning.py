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

def cols_drop(df):
    try:
        df = df.drop(columns=["Trainer", "Training Program Name",
                               "LocationCode", "Supervisor", "Location"], errors="ignore")
        return df
    except Exception as e:
        logger.exception(f"Error in cols_drop: {str(e)}")
        raise

def start_date(df):
    try:
        if "StartDate" in df.columns:
            df['StartDate'] = pd.to_datetime(df['StartDate'], errors='coerce')
        return df
    except Exception as e:
        logger.exception(f"Error in start_date: {str(e)}")
        raise

def title(df):
    try:
        if 'Title' not in df.columns:
            logger.warning("Title column not found, skipping title processing")
            return df

        df['Title'] = df['Title'].astype(str).str.strip().str.title()
        df['Title'] = df['Title'].str.replace(r'Sr\.', 'Senior', regex=True)
        df['Title'] = df['Title'].str.replace(r'Sr', 'Senior', regex=True)

        def categorize(title):
            title = str(title).lower()
            if 'engineer' in title:
                return 'Engineering'
            elif 'data' in title or 'dba' in title:
                return 'Data'
            elif 'sales' in title:
                return 'Sales'
            elif 'it' in title:
                return 'IT'
            elif 'accountant' in title:
                return 'Finance'
            else:
                return 'Other'

        df['Title_Category'] = df['Title'].apply(categorize)
        df.drop(columns=['Title'], inplace=True, errors='ignore')
        return df

    except Exception as e:
        logger.exception(f"Error in title: {str(e)}")
        raise

def Department(df):
    try:
        if 'DepartmentType' in df.columns:
            df['DepartmentType'] = df['DepartmentType'].astype(str).str.strip()
        return df
    except Exception as e:
        logger.exception(f"Error in Department: {str(e)}")
        raise

def division(df):
    try:
        if 'Division' not in df.columns:
            return df

        df['Division'] = df['Division'].astype(str).str.strip()
        df['Division'] = df['Division'].str.replace(' / ', ' & ', regex=False)

        def map_division(div):
            div = str(div).lower()
            if 'finance' in div or 'accounting' in div:
                return 'Finance'
            elif 'it' in div or 'technology' in div:
                return 'IT'
            elif 'sales' in div or 'marketing' in div:
                return 'Sales'
            elif 'project' in div:
                return 'Project Management'
            elif 'field' in div or 'construction' in div or 'wireline' in div:
                return 'Operations'
            elif 'engineer' in div:
                return 'Engineering'
            elif 'executive' in div:
                return 'Executive'
            else:
                return 'Other'

        df['Division_Category'] = df['Division'].apply(map_division)
        df.drop(columns=['Division'], inplace=True, errors='ignore')
        return df

    except Exception as e:
        logger.exception(f"Error in division: {str(e)}")
        raise

def JobFunction(df):
    try:
        if 'JobFunctionDescription' not in df.columns:
            return df

        df['JobFunctionDescription'] = df['JobFunctionDescription'].astype(str).str.strip()

        def map_job_function(job):
            job = str(job).lower()
            if 'engineer' in job or 'technician' in job or 'splicer' in job:
                return 'Technical'
            elif 'manager' in job or 'director' in job or 'supervisor' in job:
                return 'Management'
            elif 'ceo' in job or 'cfo' in job or 'cio' in job or 'evp' in job or 'svp' in job:
                return 'Executive'
            elif 'assistant' in job or 'clerk' in job or 'administrative' in job:
                return 'Admin'
            elif 'analyst' in job or 'specialist' in job or 'coordinator' in job:
                return 'Professional'
            elif 'labor' in job or 'operator' in job or 'helper' in job:
                return 'Labor'
            elif 'account' in job or 'finance' in job:
                return 'Finance'
            elif 'driver' in job or 'mechanic' in job or 'warehouse' in job:
                return 'Operations'
            else:
                return 'Other'

        df['JobFunction_Category'] = df['JobFunctionDescription'].apply(map_job_function)
        df.drop(columns=['JobFunctionDescription'], inplace=True, errors='ignore')
        return df

    except Exception as e:
        logger.exception(f"Error in JobFunction: {str(e)}")
        raise

def training(df):
    try:
        if 'Training Date' in df.columns:
            df['Training Date'] = pd.to_datetime(df['Training Date'], errors='coerce')
        return df
    except Exception as e:
        logger.exception(f"Error in training: {str(e)}")
        raise

def save_data(data_path, df):
    try:
        save_path = os.path.join(data_path, "interim")
        os.makedirs(save_path, exist_ok=True)
        output_file = os.path.join(save_path, "df_clean.csv")
        df.to_csv(output_file, index=False)
        logger.info(f"Data saved locally at: {output_file}")
        return output_file                                  
    except Exception as e:
        logger.exception(f"Error in save_data: {str(e)}")
        raise

# def download_from_s3(bucket_name, s3_key, local_path):
#     try:
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
        logger.info("Starting Data Preprocessing Pipeline...")
        config = load_config()["data_preprocessing"]           

        # local_input = "data/raw/df_gath.csv"

        # download_from_s3(
        #     bucket_name=config["bucket_name"],
        #     s3_key=config["input_s3_key"],
        #     local_path=local_input
        # )

        local_input = "data/raw/df_gath.csv"  # use local file instead

        df = data_load(local_input)
        df = start_date(df)
        df = title(df)
        df = cols_drop(df)
        df = Department(df)
        df = division(df)
        df = JobFunction(df)
        df = training(df)

        local_file = save_data(config["local_save_path"], df)  

        # upload_to_s3(
        #     local_file=local_file,
        #     bucket_name=config["bucket_name"],
        #     s3_key=config["output_s3_key"]
        # )

        logger.info("Data Preprocessing Pipeline completed successfully.")

    except Exception as e:
        logger.critical(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()