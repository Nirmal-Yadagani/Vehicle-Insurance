import sys
from src.exception import MyException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.entity.config_entity import DataIngestionConfig, DataValidationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact


class TrainPipeline:
    """"""
    def __init__(self):
        """"""
        # Load the configuration settings for data ingestion (e.g., paths, database details)
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()


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
        

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        """
        Triggers the data validation component to verify the integrity of ingested data.

        This method acts as a bridge, taking the output (DataIngestionArtifact) from the data ingestion 
        stage and passing it to the validation logic to ensure schema consistency.

        Args:
            data_ingestion_artifact (DataIngestionArtifact): Contains the file paths of 
                the training and testing datasets generated during ingestion.

        Returns:
            DataValidationArtifact: Metadata containing the validation status and the 
                path to the generated validation report.

        Raises:
            MyException: If any error occurs during the initialization or execution 
                of the validation process.
        """
        logging.info("Entered the start_data_validation method of TrainPipeline class")
        try:
            
            # Initialize the DataValidation component with the predefined config
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact, data_validation_config=self.data_validation_config)

            data_validation_artifact = data_validation.initiate_data_validation()

            logging.info("Performed the data validation operation")
            logging.info("Exited the start_data_validation method of TrainPipeline class")
            return data_validation_artifact
        
        except Exception as e:
            raise MyException(e, sys)
        

    def run_pipeline(self) -> None:
        """"""

        try:
            # Start Ingestion, this captures the artifact which will eventually be passed to Data Validation
            data_ingestion = self.start_data_ingestion()
            data_validation = self.start_data_validation(data_ingestion_artifact=data_ingestion)
            
        except Exception as e:
            raise MyException(e, sys)

