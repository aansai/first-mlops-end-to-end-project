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
    try:
        df = pd.read_csv(url)
        logger.info("Data load Successfully")
        return df
    except Exception as e:
        logger.exception(f"Error in data_load: {str(e)}")
        raise

def save_data(data_path,df):
    try:
        save_path = os.path.join(data_path,"external")
        os.makedirs(save_path,exist_ok=True)
        df.to_csv(os.path.join(save_path,"df_gath.csv"),index=False)
        logger.info("Data saved successfully")
    except Exception as e:
        logger.exception(f"Error in save_data: {str(e)}")
        raise

def main():
    try:
        df = data_load(r'C:\1ML\Messy_HR_Dataset_Detailed.csv')
        save_data("data",df)
    except Exception as e:
        logger.critical(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
     main()