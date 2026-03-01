import logging
import sys
from datetime import datetime

from src.exception import MyException
from src.logger import logger
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationAritfact, ModelTrainerArtifact


class TrainPipeline:
    """"""
    def __init__(self):
        """"""
        try:
            # Generate a unique run ID based on the current timestamp to track this specific execution of the pipeline
            self.run_id = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

            self.log = logger.bind(run_id=self.run_id)

            # Load the configuration settings for data ingestion (e.g., paths, database details)
            self.data_ingestion_config = DataIngestionConfig()
            self.data_validation_config = DataValidationConfig()
            self.data_transformation_config = DataTransformationConfig()
            self.model_trainer_config = ModelTrainerConfig()
        
        except Exception as e:
            custom_error = e if isinstance(e, MyException) else MyException(e, sys)
            logger.error("Error initializing TrainPipeline", error=str(custom_error))
            raise custom_error


    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        Triggers the data ingestion component to fetch and split data.

        Returns:
            DataIngestionArtifact: Metadata containing paths to the exported train and test files.
        
        Raises:
            MyException: If any error occurs during the ingestion process.
        """
        try:
            self.log.info('pipeline_stage_started', stage_name="Data Ingestion")

            # Initialize the DataIngestion component with the predefined config
            data_ingestion = DataIngestion(self.data_ingestion_config)

            # Execute the ingestion process and get back the artifact (file paths)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

            self.log.info('pipeline_stage_completed', stage_name="Data Ingestion", train_path=data_ingestion_artifact.trained_file_path, test_path=data_ingestion_artifact.test_file_path)
            return data_ingestion_artifact
        
        except Exception as e:
            custom_error = e if isinstance(e, MyException) else MyException(e, sys)
            logger.error("pipeline_stage_failed", stage_name="Data Ingestion", error=str(custom_error))
            raise custom_error
        

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
        try:
            self.log.info('pipeline_stage_started', stage_name="Data Validation")

            # Initialize the DataValidation component with the predefined config
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact, data_validation_config=self.data_validation_config)

            data_validation_artifact = data_validation.initiate_data_validation()

            self.log.info('pipeline_stage_completed', stage_name="Data Validation", validation_status=data_validation_artifact.validation_status, report_path=data_validation_artifact.validation_report_file_path)
            return data_validation_artifact
        
        except Exception as e:
            custom_error = e if isinstance(e, MyException) else MyException(e, sys)
            logger.error("pipeline_stage_failed", stage_name="Data Validation", error=str(custom_error))
            raise custom_error
        
    
    def start_data_transformation(self, data_ingestion_artifact: DataIngestionArtifact, 
                                  data_validation_artifact: DataValidationArtifact) -> DataTransformationAritfact:
        try:
            self.log.info('pipeline_stage_started', stage_name="Data Transformation")
            # Initialize the DataTransformation component with the predefined config
            data_transformation = DataTransformation(data_ingestion_artifact=data_ingestion_artifact,
                                                     data_validation_artifact=data_validation_artifact,
                                                     data_transformation_config=self.data_transformation_config)
            
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            self.log.info('pipeline_stage_completed', stage_name="Data Transformation")
            return data_transformation_artifact
            

        except Exception as e:
            custom_error = e if isinstance(e, MyException) else MyException(e, sys)
            logger.error("pipeline_stage_failed", stage_name="Data Transformation", error=str(custom_error))
            raise custom_error
        
    
    def start_model_trainer(self, data_transformation_artifact: DataIngestionArtifact) -> ModelTrainerArtifact:
        try:
            self.log.info('pipeline_stage_started', stage_name="Model Trainer")
            model_trainer = ModelTrainer(data_transformation_artifact=data_transformation_artifact,
                                         model_trainer_config=self.model_trainer_config)
            
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            self.log.info('pipeline_stage_completed', stage_name="Model Trainer")
            return model_trainer_artifact

        except Exception as e:
            custom_error = e if isinstance(e, MyException) else MyException(e, sys)
            self.log.error("pipeline_stage_failed", stage_name="Model Trainer", error=str(custom_error))
            raise custom_error
        

    def run_pipeline(self) -> None:
        """"""

        try:
            self.log.info("entire_pipeline_run_initiated")
            # Start Ingestion, this captures the artifact which will eventually be passed to Data Validation
            data_ingestion = self.start_data_ingestion()
            data_validation = self.start_data_validation(data_ingestion_artifact=data_ingestion)
            data_transformation = self.start_data_transformation(data_ingestion_artifact=data_ingestion,
                                                                 data_validation_artifact=data_validation)
            model_trainer = self.start_model_trainer(data_transformation_artifact=data_transformation)
            self.log.info("entire_pipeline_run_completed", final_model_path=model_trainer.trained_model_file_path)
            
        except Exception as e:
            custom_error = e if isinstance(e, MyException) else MyException(e, sys)
            logger.error("entire_pipeline_run_failed", error=str(custom_error))
            raise custom_error

