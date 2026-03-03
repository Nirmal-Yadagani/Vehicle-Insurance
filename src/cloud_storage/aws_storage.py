import os
import sys
from io import StringIO
from typing import Union, List

import pickle
from mypy_boto3_s3.service_resource import Bucket


from src.configuration.aws_connection import S3Client
from src.logger import logger
from src.exception import MyException


class SimpleStorageService:
    """
    A utility class to interact with AWS S3, providing methods for bucket access, 
    file checking, model loading, and file uploads.
    """
    def __init__(self):
        """
        Initializes the S3 client and resource.
        
        Raises:
            MyException: If connection to AWS fails.
        """
        try:
            s3_client = S3Client()
            self.s3_resource = s3_client.s3_resourse
            self.s3_client = s3_client.s3_client
            self.log = logger.bind(service='S3Storage')

        except Exception as e:
            custom_error = e if isinstance(e, MyException) else MyException(e, sys)
            logger.error('s3 connection failed',error=str(custom_error))
            raise custom_error

    
    def s3_key_path_avilable(self, bucket_name, s3_key) -> bool:
        """
        Checks if a specific S3 key (file/folder) exists in the given bucket.

        Args:
            bucket_name (str): Name of the S3 bucket.
            s3_key (str): The path/key to look for.

        Returns:
            bool: True if path exists, False otherwise.
        """
        try:
            bucket = self.get_bucket(bucket_name)
            file_objects = [file_object for file_object in bucket.objects.filter(Prefix=s3_key)]
            status = len(file_objects) > 0
            self.log.info('s3 key check',bucket=bucket_name, key=s3_key, exists=status)
            return status
        
        except Exception as e:
            custom_error = e if isinstance(e, MyException) else MyException(e, sys)
            self.log.error('s3 key check failed', bucket=bucket_name, key=s3_key, error=str(custom_error))
            raise custom_error


    @staticmethod
    def read_object(object_name: str, decode: bool = True, make_redable: bool = False) -> Union[StringIO, str]:
        """
        Reads the body of an S3 object.

        Args:
            object_name (object): The S3 object instance.
            decode (bool): Whether to decode the bytes to string. Defaults to True.
            make_redable (bool): Whether to wrap output in StringIO. Defaults to False.

        Returns:
            Union[StringIO, str, bytes]: The content of the S3 object.
        """
        try:
            func = (lambda: object_name.get()['Body'].read().decode() if decode else object_name.get()['Body'].read())

            conv_func = lambda: StringIO(func()) if make_redable else func()

            return conv_func()

        except Exception as e:
            if isinstance(e, MyException):
                raise e
            raise MyException(e, sys) 
        

    def get_bucket(self, bucket_name: str) -> Bucket:
        try:
            bucket = self.s3_resource.Bucket(bucket_name)
            return bucket
        
        except Exception as e:
            if isinstance(e, MyException):
                raise e
            raise MyException(e, sys)


    def get_file_object(self, filename: str, bucket_name: str) -> Union[List[object], object]:
        """
        Fetches file object(s) matching a specific prefix from S3.

        Args:
            filename (str): The filename or prefix to search.
            bucket_name (str): The bucket to search in.

        Returns:
            Union[List[object], object]: A single S3 object or a list of objects if multiple found.
        """
        try:
            bucket = self.get_bucket(bucket_name=bucket_name)
            file_objects = [file_object for file_object in bucket.objects.filter(Prefix=filename)]

            if len(file_objects) == 0:
                self.log.warning('s3 file not found', filename=filename, bucket=bucket_name)
                return None

            return file_objects[0] if len(file_objects) == 1 else file_objects
        
        except Exception as e:
            if isinstance(e, MyException):
                raise e
            raise MyException(e, sys)


    def load_model(self, model_name: str, bucker_name: str, model_dir: str = None) -> object:
        """
        Downloads and deserializes a pickle model from S3.

        Args:
            model_name (str): Name of the model file.
            bucker_name (str): The S3 bucket name.
            model_dir (str, optional): The directory inside the bucket. Defaults to None.

        Returns:
            object: The loaded Python object (model).
        """
        try:
            model_file = f"{model_dir}/{model_name}" if model_dir else model_name
            self.log.info('loading model from s3', path = model_file, bucker=bucker_name)
            file_object = self.get_file_object(model_file, bucket_name=bucker_name)

            if file_object is None:
                raise Exception(f"Model file {model_file} not found in bucket {bucker_name}")

            model_obj = self.read_object(file_object, decode=False)
            model = pickle.loads(model_obj)
            self.log.info("model loaded successfully", model_type=type(model).__name__)
            return model
        
        except Exception as e:
            custom_error = e if isinstance(e, MyException) else MyException(e, sys)
            self.log.error("model_load_failed", error=str(custom_error))
            raise custom_error
        

    def upload_file(self, from_filename:str, to_filename:str, bucket_name:str, remove:bool = True):
        """
        Uploads a local file to an S3 bucket.

        Args:
            from_filename (str): Local file path.
            to_filename (str): Desired S3 key path.
            bucket_name (str): Destination S3 bucket.
            remove (bool): If True, deletes the local file after upload. Defaults to True.
        """
        try:
            self.log.info("uploading_file_to_s3", local=from_filename, remote=to_filename, bucket=bucket_name)
            self.s3_resource.meta.client.upload_file(from_filename, bucket_name, to_filename)

            if remove:
                os.remove(from_filename)
                self.log.info("local file removed", path=from_filename)

        except Exception as e:
            custom_error = e if isinstance(e, MyException) else MyException(e, sys)
            self.log.error("s3_upload_failed", error=str(custom_error))
            raise custom_error
        

    