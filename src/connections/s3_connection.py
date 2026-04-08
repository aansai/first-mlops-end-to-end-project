import os
import boto3
import logging
from botocore.exceptions import ClientError


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class S3Connection:
    def __init__(self, region_name=None):
        self.region_name = region_name or os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        self.client = boto3.client("s3", region_name=self.region_name)

    def upload_file(self, file_path, bucket_name, object_name=None):
        object_name = object_name or os.path.basename(file_path)

        try:
            self.client.upload_file(file_path, bucket_name, object_name)
            logger.info(f"Uploaded {object_name} to {bucket_name}")
            return True
        except ClientError as e:
            logger.error(f"Upload failed: {e}")
            raise

    def download_file(self, bucket_name, object_name, file_path):
        try:
            self.client.download_file(bucket_name, object_name, file_path)
            logger.info(f"Downloaded {object_name} from {bucket_name}")
            return True
        except ClientError as e:
            logger.error(f"Download failed: {e}")
            raise

    def list_files(self, bucket_name):
        try:
            response = self.client.list_objects_v2(Bucket=bucket_name)
            return [obj["Key"] for obj in response.get("Contents", [])]
        except ClientError as e:
            logger.error(f"List failed: {e}")
            raise

    def delete_file(self, bucket_name, object_name):
        try:
            self.client.delete_object(Bucket=bucket_name, Key=object_name)
            logger.info(f"Deleted {object_name} from {bucket_name}")
            return True
        except ClientError as e:
            logger.error(f"Delete failed: {e}")
            raise