import os
import sys

import boto3
from dotenv import load_dotenv

from src.constants import AWS_ACCESS_KEY_ID_ENV_KEY, AWS_SECRET_ACCESS_KEY_ENV_KEY, REGION_NAME
from src.logger import logger
from src.exception import MyException


class S3Client:
    """
    A Singleton class to manage AWS S3 connections.
    Ensures that only one instance of the S3 client and resource is created 
    throughout the application lifecycle.
    """
    s3_client = None
    s3_resourse = None
    def __init__(self, region_name = REGION_NAME):
        """
        Initializes the S3 client and resource using environment variables.

        Args:
            region_name (str): The AWS region to connect to. Defaults to REGION_NAME.
        
        Raises:
            MyException: If environment variables are missing or connection fails.
        """
        try:
            if not S3Client.s3_client or not S3Client.s3_resourse:
                load_dotenv()
                __access_key_id = os.getenv(AWS_ACCESS_KEY_ID_ENV_KEY)
                __secret_access_key = os.getenv(AWS_SECRET_ACCESS_KEY_ENV_KEY)

                if not __access_key_id:
                    logger.error("aws_credential_missing", env_key=AWS_ACCESS_KEY_ID_ENV_KEY)
                    raise Exception(f"Environment variable: {AWS_ACCESS_KEY_ID_ENV_KEY} is not not set.")
                
                if not __secret_access_key:
                    logger.error("aws_credential_missing", env_key=AWS_SECRET_ACCESS_KEY_ENV_KEY)
                    raise Exception(f"Environment variable: {AWS_SECRET_ACCESS_KEY_ENV_KEY} is not set.")
                
                S3Client.s3_resourse = boto3.resource('s3', aws_access_key_id=__access_key_id,
                                                    aws_secret_access_key=__secret_access_key,
                                                    region_name=region_name)

                S3Client.s3_client = boto3.client('s3', aws_access_key_id=__access_key_id,
                                                aws_secret_access_key=__secret_access_key,
                                                region_name=region_name)
                logger.info("aws_s3_connection_established", region=region_name)


            self.s3_resourse = S3Client.s3_resourse
            self.s3_client = S3Client.s3_client

        except Exception as e:
            custom_error = e if isinstance(e, MyException) else MyException(e, sys)
            # Log the error but ensure no sensitive data is leaked
            logger.error("aws_s3_connection_failed", error=str(e))
            raise custom_error
