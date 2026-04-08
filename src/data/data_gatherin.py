import yaml
import pandas as pd
import os
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

def save_data(data_path, df):
    try:
        save_path = os.path.join(data_path, "raw")
        os.makedirs(save_path, exist_ok=True)
        output_file = os.path.join(save_path, "df_gath.csv")
        df.to_csv(output_file, index=False)
        logger.info(f"Data saved locally at: {output_file}")
        return output_file                   
    except Exception as e:
        logger.exception(f"Error in save_data: {str(e)}")
        raise

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
        logger.info("Starting Data Gathering Pipeline...")
        config = load_config()["data_gathering"]     
        df = data_load(config["source_path"])       
        local_file = save_data(config["local_save_path"], df) 

        # upload_to_s3(  # COMMENTED OUT
        #     local_file=local_file,
        #     bucket_name=config["bucket_name"],
        #     s3_key=config["s3_key"]
        # )

        logger.info("Data Gathering Pipeline completed successfully.")

    except Exception as e:
        logger.critical(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()