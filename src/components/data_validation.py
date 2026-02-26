import json
import sys
import os

import pandas as pd
from pandas import DataFrame

from src.exception import MyException
from src.logger import logging
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

        except Exception as e:
            raise MyException(e, sys)
        

    def validate_number_of_columns(self, dataframe: DataFrame) -> bool:
        """
        Validates if the total number of columns in the dataframe matches the schema.

        Args:
            dataframe: The pandas DataFrame to check.

        Returns:
            bool: True if column count matches, False otherwise.
        """
        try:
            status = len(dataframe.columns) == len(self._schema_config['columns'])
            logging.info(f"Does number of columns match: [{status}]")
            return status

        except Exception as e:
            raise MyException(e, sys)
        
    
    def is_column_exist(self, dataframe: DataFrame) -> bool:
        """
        Checks if all required numerical and categorical columns exist in the dataframe.

        Args:
            dataframe: The pandas DataFrame to verify against schema.

        Returns:
            bool: True if all required columns are present, False if any are missing.
        """
        try:
            dataframe_columns = dataframe.columns
            missing_numerical_columns = []
            missing_categorical_columns = []
            for column in self._schema_config['numerical_columns']:
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)


            for column in self._schema_config['categorical_columns']:
                if column not in dataframe_columns:
                    missing_categorical_columns.append(column)

            if len(missing_numerical_columns) > 0:
                logging.info(f"Missing there numerical columns: {missing_numerical_columns}")

            if len(missing_categorical_columns) > 0:
                logging.info(f"Missing there categorical columns: {missing_categorical_columns}")

            return False if len(missing_categorical_columns)>0 or len(missing_numerical_columns)>0 else True
        
        except Exception as e:
            raise MyException(e, sys)
        
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
            raise MyException(e, sys)
        

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Executes the data validation suite and generates a validation report.

        Returns:
            DataValidationArtifact: Object containing validation status and report path.

        Raises:
            MyException: If validation process fails.
        """
        try:
            validation_error_msg = ""
            logging.info("Starting data validation")
            train_df, test_df = (DataValidation.read_data(file_path=self.data_ingestion_artifact.trained_file_path),
                                 DataValidation.read_data(file_path=self.data_ingestion_artifact.test_file_path))
            
            status = self.validate_number_of_columns(dataframe=train_df)
            if not status:
                validation_error_msg += "Columns are missing in training dataframe."
            else:
                logging.info(f"All required columns present in training dataframe: [{status}]")


            status = self.is_column_exist(dataframe=train_df)
            if not status:
                validation_error_msg += "Columns are missing in training dataframe."
            else:
                logging.info(f"All categorical/int columns present in training dataframe: [{status}]")


            status = self.validate_number_of_columns(dataframe=test_df)
            if not status:
                validation_error_msg += "Columns are missing in testing dataframe."
            else:
                logging.info(f"All required columns present in testing dataframe: {status}")


            status = self.is_column_exist(dataframe=test_df)
            if not status:
                validation_error_msg += "Columns are missing in testing dataframe."
            else:
                logging.info(f"All categorical/int columns present in testing dataframe: {status}")
            

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
            
            logging.info("Data validation artifact created and saved to JSON file.")
            logging.info(f"Data Validation Artifact: {data_validation_artifact}")
            return data_validation_artifact
        
        except Exception as e:
            raise MyException(e, sys)
            

