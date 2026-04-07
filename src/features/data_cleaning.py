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

def cols_drop(df):
    try:
        df = df.drop(columns=["Trainer","Training Program Name",
        "LocationCode","Supervisor","Location"], errors="ignore")
        return df
    except Exception as e:
        logger.exception("Error in cols_drop")
        raise

def start_date(df):
    try:
        df['StartDate'] = pd.to_datetime(df['StartDate'],format='%d-%b-%y')
        return df
    except Exception as e:
        logger.exception("Error in start_date")
        raise

def title(df):
    try:
        if 'Title' not in df.columns:
            logger.warning("Title column not found, skipping title processing")
            return df

        df['Title'] = df['Title'].str.strip()
        df['Title'] = df['Title'].str.title()
        df['Title'] = df['Title'].str.replace('Sr\\.', 'Senior', regex=True)
        df['Title'] = df['Title'].str.replace('Sr', 'Senior', regex=True)

        def categorize(title):
            title = title.lower()
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
        logger.exception("Error in title")
        raise

def Department(df):
    try:
        df['DepartmentType'] = df['DepartmentType'].str.strip()
        return df
    except Exception as e:
        logger.exception("Error in Department")
        raise

def division(df):
    try:
        df['Division'] = df['Division'].str.strip()
        df['Division'] = df['Division'].str.replace(' / ', ' & ', regex=False)

        def map_division(div):
            div = div.lower()
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
        df.drop(columns=['Division'],inplace=True)
        return df
    except Exception as e:
        logger.exception("Error in division")
        raise

def JobFunction(df):
    try:
        df['JobFunctionDescription']  = df['JobFunctionDescription'].str.strip()

        def map_job_function(job):
            job = job.lower()

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
        df['JobFunction_Category'].unique()
        df.drop(columns=['JobFunctionDescription'],inplace=True)
        return df
    except Exception as e:
        logger.exception("Error in JobFunction")
        raise

def trainiing(df):
    try:
        df['Training Date'] = pd.to_datetime(df['Training Date'],format='%d-%b-%y')
        return df 
    except Exception as e:
        logger.exception("Error in trainiing")
        raise

def save_data(data_path,df):
    try:
        save_path = os.path.join(data_path,"interim")
        os.makedirs(save_path,exist_ok=True)
        df.to_csv(os.path.join(save_path,"df_clean.csv"),index=False)
        logger.info("Data saved successfully")
    except Exception as e:
        logger.exception(f"Error in save_data: {str(e)}")
        raise

def main():
    try:
        df = data_load(r'C:\1ML\data\external\df_gath.csv')
        df = start_date(df)
        df = title(df)
        df = cols_drop(df)
        df = Department(df)
        df = division(df)
        df = JobFunction(df)
        df = trainiing(df)
        save_data(r'C:\1ML\data', df)
    except Exception as e:
        logger.exception("Error in main")
        raise

if __name__ == "__main__":
     main()

