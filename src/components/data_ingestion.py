import os
import sys

from pandas import DataFrame
from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exception import MyException
from src.data_access.data_access import DataAccess
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact


class DataIngestion:
    """
    Handles the data ingestion pipeline, including fetching raw data from MongoDB,
    storing it in a local feature store, and splitting it into train and test sets.
    """
    def __init__(self, data_ingestion_config: DataIngestionConfig=DataIngestionConfig()):
        """
        Initializes the DataIngestion component with configuration settings.
        
        Args:
            data_ingestion_config: Configuration object containing file paths and split ratios.
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise MyException(e, sys)
        

    def export_data_into_feature_store(self) -> DataFrame:
        """
        Fetches data from the MongoDB collection and saves it as a CSV in the feature store.

        Returns:
            DataFrame: The raw data exported from the database.
        
        Raises:
            MyException: If any error occurs during database access or file writing.
        """
        try:
            logging.info(f"Exporting data from mongodb")
            # Initialize data access object to interact with MongoDB
            data = DataAccess()
            dataframe = data.export_collection_as_dataframe(collection_name=self.data_ingestion_config.collection_name)
            logging.info(f"Shape of dataframe: {dataframe.shape}")

            # Define path and ensure the directory exists before saving
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)

            # Save the raw dataframe to the feature store (local CSV)
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            logging.info(f"Exported data into feature store file path: {feature_store_file_path}")
            return dataframe
        except Exception as e:
            raise MyException(e ,sys)
        
    
    def split_data_as_train_test(self, dataframe: DataFrame) -> None:
        """
        Splits the provided dataframe into training and testing sets based on the config ratio.

        Args:
            dataframe: The pandas DataFrame to be split.
        
        Raises:
            MyException: If the split or file writing process fails.
        """
        logging.info("Entered split_data_as_train_test method of Data_Ingestion class")
        try:
            # Perform the split using sklearn.train_test_split
            train_set, test_set = train_test_split(dataframe, test_size=self.data_ingestion_config.train_test_split_ratio)
            logging.info("Performed train test split on dataframe")
            logging.info("Exited split_data_ad_train_test method of Data_ingestion class")

            # Prepare the directory for training/testing files
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            logging.info(f"Exporting data to train and test file path")
            # Export train and test sets to CSV
            train_set.to_csv(self.data_ingestion_config.training_file_path)
            test_set.to_csv(self.data_ingestion_config.testing_file_path)
            logging.info("Expoted train and test data to respective file path.")
        except Exception as e:
            raise MyException(e, sys)
        

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Orchestrates the data ingestion process: exports data, splits it, and returns artifacts.

        Returns:
            DataIngestionArtifact: Contains the file paths for the generated train and test datasets.
            
        Raises:
            MyException: If any step in the ingestion pipeline fails.
        """
        logging.info("Entered initiate_data_ingestion method of DataIngestion class")
        try:
            # Pull data from source
            dataframe = self.export_data_into_feature_store()

            logging.info('Got the data from mongoDB')
            # Split data for model training
            self.split_data_as_train_test(dataframe=dataframe)

            logging.info('Performed train test split on the dataset')
            logging.info('Exited initiate_data_ingestion method of DataIngestion class')

            # Package the results into an artifact object
            data_ingestion_artifact = DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path, test_file_path=self.data_ingestion_config.testing_file_path)
            logging.info(f"Data Ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise MyException(e, sys)

