import sys
from src.exception import MyException
from src.logger import logging

from src.components.data_ingestion import DataIngestion


from src.entity.config_entity import DataIngestionConfig

from src.entity.artifact_entity import DataIngestionArtifact

class TrainPipeline:
    """"""
    def __init__(self):
        """"""
        # Load the configuration settings for data ingestion (e.g., paths, database details)
        self.data_ingestion_config = DataIngestionConfig()



    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        Triggers the data ingestion component to fetch and split data.

        Returns:
            DataIngestionArtifact: Metadata containing paths to the exported train and test files.
        
        Raises:
            MyException: If any error occurs during the ingestion process.
        """
        try:
            logging.info("Entered the start_data_ingestion method of TrainPipeline class")
            logging.info("Getting the data from mongodb")

            # Initialize the DataIngestion component with the predefined config
            data_ingestion = DataIngestion(self.data_ingestion_config)

            # Execute the ingestion process and get back the artifact (file paths)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

            logging.info("Got the train_set and test_set from mongodb")
            logging.info("Exited the start_data_ingestion method of TrainPipeline class")
            return data_ingestion_artifact
        except Exception as e:
            raise MyException(e, sys)
        

    def run_pipeline(self) -> None:
        """"""

        try:
            # Start Ingestion, this captures the artifact which will eventually be passed to Data Validation
            data_ingestion = self.start_data_ingestion()
            
        except Exception as e:
            raise MyException(e, sys)

