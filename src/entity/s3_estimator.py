import sys

from pandas import DataFrame

from src.cloud_storage.aws_storage import SimpleStorageService
from src.exception import MyException
from src.logger import logger
from src.entity.estimator import MyModel



class S3Estimator:
    """
    Handles interactions with models stored in AWS S3.
    Provides a seamless interface to check, load, save, and run predictions 
    directly using S3-hosted model artifacts.
    """
    def __init__(self, bucker_name, model_path):
        """
        Initializes the S3Estimator.

        Args:
            bucker_name (str): Name of the S3 bucket.
            model_path (str): S3 key/path to the model file.
        """
        self.bucket_name = bucker_name
        self.s3 = SimpleStorageService()
        self.model_path = model_path
        self.loaded_model: MyModel = None
        self.log = logger.bind(bucket=bucker_name, s3_path=model_path)

    
    def is_model_present(self, model_path):
        """
        Verifies if the model exists in the specified S3 path.

        Args:
            model_path (str): The S3 key to check.

        Returns:
            bool: True if model exists, False otherwise.
        """
        try:
            status = self.s3.s3_key_path_avilable(self.bucket_name, model_path)
            self.log.info("check_model_existence", exists=status)
            return status
        
        except Exception as e:
            custom_error = e if isinstance(e, MyException) else MyException(e, sys)
            self.log.error("error_checking_model_presence", error=str(custom_error))
            return False
        

    def load_model(self) -> MyModel:
        """
        Downloads the model from S3 and loads it into memory.

        Returns:
            MyModel: The loaded model object containing preprocessing and estimator.
        """
        try:
            self.log.info("loading_model_into_memory")
            model = self.s3.load_model(self.model_path, self.bucket_name)
            return model
        
        except Exception as e:
            custom_error = e if isinstance(e, MyException) else MyException(e, sys)
            self.log.error("failed_to_load_model_from_s3", error=str(custom_error))
            raise custom_error
    

    def save_model(self, from_file, remove:bool=False) -> None:
        """
        Uploads a local model file to S3.

        Args:
            from_file (str): Local path to the model file.
            remove (bool): If True, deletes the local file after upload. Defaults to False.
        """
        try:
            self.log.info("saving_model_to_s3", local_source=from_file)
            self.s3.upload_file(
                from_filename=from_file, 
                to_filename=self.model_path, 
                bucket_name=self.bucket_name, 
                remove=remove
            )
            self.log.info("model_saved_successfully")

        except Exception as e:
            custom_error = e if isinstance(e, MyException) else MyException(e, sys)
            self.log.error("failed_to_save_model_to_s3", error=str(custom_error))
            raise custom_error
        

    def predict(self, dataframe:DataFrame):
        """
        Performs inference by lazily loading the model from S3 if not already in memory.

        Args:
            dataframe (DataFrame): Input data for prediction.

        Returns:
            DataFrame: Prediction results.
        """
        try:
            if self.loaded_model is None:
                self.log.info("model_not_in_memory_initializing_lazy_load")
                self.loaded_model = self.load_model()
            
            return self.loaded_model.predict(dataframe)
        
        except Exception as e:
            custom_error = e if isinstance(e, MyException) else MyException(e, sys)
            self.log.error("s3_prediction_failed", error=str(custom_error))
            raise custom_error
            
