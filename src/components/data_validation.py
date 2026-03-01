import json
import os
import sys

import pandas as pd
from pandas import DataFrame

from src.exception import MyException
from src.logger import logger
from src.utils.main_utils  import read_yaml_file
from src.entity.config_entity import DataValidationConfig
from src.entity.artifact_entity import DataValidationArtifact, DataIngestionArtifact
from src.constants import SCHEMA_FILE_PATH


class DataValidation:
    """
    Performs validation on the ingested data by checking schema consistency, 
    column presence, and data integrity for both train and test sets.
    """
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        """
        Initializes DataValidation with ingestion artifacts and validation configurations.

        Args:
            data_ingestion_artifact: Artifact containing paths to the ingested train and test data.
            data_validation_config: Configuration for validation reports and thresholds.
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
            self.log = logger.bind(report_path=self.data_validation_config.validation_report_file_path)

        except Exception as e:
            custom_error = e if isinstance(e, MyException) else MyException(e, sys)
            logger.error(f"Error initializing DataValidation", error=str(custom_error))
            raise custom_error
        

    def validate_number_of_columns(self, dataframe: DataFrame, dataset_type: str) -> bool:
        """
        Validates if the total number of columns in the dataframe matches the schema.

        Args:
            dataframe: The pandas DataFrame to check.

        Returns:
            bool: True if column count matches, False otherwise.
        """
        try:
            expected_count = len(self._schema_config['columns'])
            actual_count = len(dataframe.columns)
            status = actual_count == expected_count

            self.log.info(f"Column count validation", dataframe_type=dataset_type, expected_count=expected_count, actual_count=actual_count, status=status)
            return status

        except Exception as e:
            custom_error = e if isinstance(e, MyException) else MyException(e, sys)
            self.log.error('Error occurred while validating number of columns.', error=str(custom_error))
            raise custom_error
        
    
    def is_column_exist(self, dataframe: DataFrame, dataset_type: str) -> bool:
        """
        Checks if all required numerical and categorical columns exist in the dataframe.

        Args:
            dataframe: The pandas DataFrame to verify against schema.

        Returns:
            bool: True if all required columns are present, False if any are missing.
        """
        try:
            dataframe_columns = dataframe.columns
            missing_numerical = [col for col in self._schema_config['numerical_columns'] if col not in dataframe_columns]
            missing_categorical = [col for col in self._schema_config['categorical_columns'] if col not in dataframe_columns]

            if missing_numerical or missing_categorical:
                self.log.warning("missing_columns_detected",
                                 dataset_type=dataset_type,
                                 missing_num=missing_numerical, 
                                 missing_cat=missing_categorical)
                return False
            
            self.log.info("All required columns are present in the dataframe.", dataset_type=dataset_type)
            return True
        
        except Exception as e:
            custom_error = e if isinstance(e, MyException) else MyException(e, sys)
            self.log.error('Error occurred while checking column existence.', error=str(custom_error))
            raise custom_error
        
    @staticmethod
    def read_data(file_path) -> DataFrame:
        """
        Helper method to read a CSV file into a pandas DataFrame.

        Args:
            file_path: System path to the CSV file.

        Returns:
            DataFrame: Loaded data.
        """
        try:
            return pd.read_csv(file_path)
        
        except Exception as e:
            if isinstance(e, MyException):
                raise e
            raise MyException(e, sys)
        

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Executes the data validation suite and generates a validation report.

        Returns:
            DataValidationArtifact: Object containing validation status and report path.

        Raises:
            MyException: If validation process fails.
        """
        self.log.info("Starting data validation process.")
        try:
            validation_error_msg = ""
            train_df, test_df = (DataValidation.read_data(file_path=self.data_ingestion_artifact.trained_file_path),
                                 DataValidation.read_data(file_path=self.data_ingestion_artifact.test_file_path))
            
            # Validate Train Data
            if not self.validate_number_of_columns(train_df, "train"):
                validation_error_msg += "Column count mismatch in Train data. "
            if not self.is_column_exist(train_df, "train"):
                validation_error_msg += "Missing specific columns in Train data. "

            # Validate Test Data
            if not self.validate_number_of_columns(test_df, "test"):
                validation_error_msg += "Column count mismatch in Test data. "
            if not self.is_column_exist(test_df, "test"):
                validation_error_msg += "Missing specific columns in Test data. "

            validation_status = len(validation_error_msg) == 0

            validation_report = {"validation_status": validation_status,
                                 "message": validation_error_msg}
            
            dir_name = os.path.dirname(self.data_validation_config.validation_report_file_path)
            os.makedirs(dir_name, exist_ok=True)
            with open(self.data_validation_config.validation_report_file_path, 'w') as file_obj:
                json.dump(validation_report, file_obj, indent=4)

            data_validation_artifact = DataValidationArtifact(validation_status=validation_status,
                                                              message=validation_error_msg,
                                                              validation_report_file_path=self.data_validation_config.validation_report_file_path)
            return data_validation_artifact
        
        except Exception as e:
            custom_error = e if isinstance(e, MyException) else MyException(e, sys)
            self.log.error("Error during data validation process.", error=str(custom_error))
            raise custom_error
            

