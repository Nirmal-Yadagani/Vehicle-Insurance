import os
import sys

from pandas import DataFrame
from sklearn.model_selection import train_test_split

from src.logger import logger
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
            self.log = logger.bind(collection_name=self.data_ingestion_config.collection_name)
            
        except Exception as e:
            custom_error = e if isinstance(e, MyException) else MyException(e, sys)
            logger.error('Error occurred while initializing DataIngestion component.', error=str(custom_error))
            raise custom_error
        

    def export_data_into_feature_store(self) -> DataFrame:
        """
        Fetches data from the MongoDB collection and saves it as a CSV in the feature store.

        Returns:
            DataFrame: The raw data exported from the database.
        
        Raises:
            MyException: If any error occurs during database access or file writing.
        """
        try:
            self.log.info(f"Exporting data from mongodb")
            # Initialize data access object to interact with MongoDB
            data = DataAccess()
            dataframe = data.export_collection_as_dataframe(collection_name=self.data_ingestion_config.collection_name)
            self.log.info(f"data fetched successfully.", dataframe_shape=dataframe.shape)

            # Define path and ensure the directory exists before saving
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)

            # Save the raw dataframe to the feature store (local CSV)
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            self.log.info(f"Exported data into feature store.", file_path=feature_store_file_path)
            return dataframe
        
        except Exception as e:
            custom_error = e if isinstance(e, MyException) else MyException(e, sys)
            self.log.error('Error occurred while exporting data into feature store.', error=str(custom_error))
            raise custom_error
        
    
    def split_data_as_train_test(self, dataframe: DataFrame) -> None:
        """
        Splits the provided dataframe into training and testing sets based on the config ratio.

        Args:
            dataframe: The pandas DataFrame to be split.
        
        Raises:
            MyException: If the split or file writing process fails.
        """
        try:
            self.log.info("Entered split_data_as_train_test method of Data_Ingestion class", ratio=self.data_ingestion_config.train_test_split_ratio)
            # Perform the split using sklearn.train_test_split
            train_set, test_set = train_test_split(dataframe, test_size=self.data_ingestion_config.train_test_split_ratio)


            # Prepare the directory for training/testing files
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            # Export train and test sets to CSV
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)
            self.log.info("Exported train and test data", train_shape=train_set.shape, test_shape=test_set.shape)

        except Exception as e:
            custom_error = e if isinstance(e, MyException) else MyException(e, sys)
            self.log.error('Error occurred while splitting data into train and test sets.', error=str(custom_error))
            raise custom_error
        

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Orchestrates the data ingestion process: exports data, splits it, and returns artifacts.

        Returns:
            DataIngestionArtifact: Contains the file paths for the generated train and test datasets.
            
        Raises:
            MyException: If any step in the ingestion pipeline fails.
        """
        self.log.info("Entered initiate_data_ingestion method of DataIngestion class")
        try:
            # Pull data from source
            dataframe = self.export_data_into_feature_store()

            # Split data for model training
            self.split_data_as_train_test(dataframe=dataframe)

            # Package the results into an artifact object
            data_ingestion_artifact = DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path, test_file_path=self.data_ingestion_config.testing_file_path)
            self.log.info("Data ingestion completed successfully", data_ingestion_artifact=data_ingestion_artifact.__dict__)
            return data_ingestion_artifact
        
        except Exception as e:
            custom_error = e if isinstance(e, MyException) else MyException(e, sys)
            self.log.error('Error occurred during the data ingestion process.', error=str(custom_error))
            raise custom_error

