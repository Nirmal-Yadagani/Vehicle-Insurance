import sys
from datetime import datetime

from src.exception import MyException
from src.logger import logger
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.components.model_pusher import ModelPusher
from src.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig, ModelEvaluationConfig, ModelPusherConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationAritfact, ModelTrainerArtifact, ModelEvaluationArtifact, ModelPusherArtifact


class TrainPipeline:
    """
    Orchestrates the entire machine learning pipeline.
    Sequentially executes ingestion, validation, transformation, training, 
    evaluation, and deployment (pushing).
    """
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
            self.model_evaluation_config = ModelEvaluationConfig()
            self.model_pusher_config = ModelPusherConfig()
        
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
        """
        Triggers the data transformation component to engineer features and handle data preprocessing.

        This method takes the validated data and applies transformations like scaling and 
        synthetic oversampling (SMOTEENN) to prepare the dataset for model training.

        Args:
            data_ingestion_artifact (DataIngestionArtifact): Contains the file paths of 
                the training and testing datasets.
            data_validation_artifact (DataValidationArtifact): Contains the validation status 
                to ensure transformation is performed on valid data.

        Returns:
            DataTransformationAritfact: Metadata containing paths to the transformed 
                numpy arrays and the saved preprocessing object.

        Raises:
            MyException: If any error occurs during the feature engineering or 
                transformation process.
        """
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
        """
        Triggers the model trainer component to fit the machine learning algorithm.

        This stage uses the transformed training data to train the model and evaluates 
        it against the transformed test data to generate performance metrics.

        Args:
            data_transformation_artifact (DataTransformationAritfact): Contains paths to 
                the transformed numpy arrays (train/test) and the preprocessing object.

        Returns:
            ModelTrainerArtifact: Metadata containing the path to the trained model file 
                and the classification metric report.

        Raises:
            MyException: If any error occurs during model training or hyperparameter 
                assignment.
        """
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
        
    
    def start_model_evaluation(self, data_ingestion_artifact: DataIngestionArtifact, model_trainer_artifact: ModelTrainerArtifact) -> ModelEvaluationArtifact:
        """
        Triggers the model evaluation component to compare the new model with the production model.

        This method acts as a decision gate, evaluating the current model's performance 
        against the 'Champion' model currently stored in S3 to determine if an update is required.

        Args:
            data_ingestion_artifact (DataIngestionArtifact): Used to retrieve the original 
                test data for unbiased comparison.
            model_trainer_artifact (ModelTrainerArtifact): Contains the newly trained 
                model object and its performance scores.

        Returns:
            ModelEvaluationArtifact: Metadata indicating whether the new model is accepted 
                for production based on the defined performance threshold.

        Raises:
            MyException: If any error occurs during the model comparison or S3 
                retrieval process.
        """
        try:
            self.log.info('pipeline_stage_started', stage_name="Model Evaluation")
            model_evaluation = ModelEvaluation(data_ingestion_artifact=data_ingestion_artifact,
                                               model_trainer_artifact=model_trainer_artifact,
                                               model_eval_config=self.model_evaluation_config)
            
            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()
            self.log.info('pipeline_stage_completed', stage_name="Model Evaluation")
            return model_evaluation_artifact

        except Exception as e:
            custom_error = e if isinstance(e, MyException) else MyException(e, sys)
            self.log.error("pipeline_stage_failed", stage_name="Model Evaluation", error=str(custom_error))
            raise custom_error


    def start_model_pusher(self, model_evaluation_artifact: ModelEvaluationArtifact) -> ModelPusherArtifact:
        """
        Triggers the model pusher component to deploy the accepted model to S3.

        This final stage automates the deployment by taking the validated model 
        artifact and uploading it to the production bucket in AWS S3.

        Args:
            model_evaluation_artifact (ModelEvaluationArtifact): Contains the acceptance 
                status and paths for the validated model.

        Returns:
            ModelPusherArtifact: Metadata confirming the S3 bucket and key where 
                the model was successfully deployed.

        Raises:
            MyException: If any error occurs during the cloud upload or file 
                handling process.
        """
        try:
            self.log.info('pipeline_stage_started', stage_name="Model Pusher")
            model_pusher = ModelPusher(model_pusher_config=self.model_pusher_config,
                                       model_evaluation_artifact=model_evaluation_artifact)
            
            model_pusher_artifact = model_pusher.initiate_model_pusher()
            self.log.info('pipeline_stage_completed', stage_name="Model Pusher")
            return model_pusher_artifact
        
        except Exception as e:
            custom_error = e if isinstance(e, MyException) else MyException(e, sys)
            self.log.error("pipeline_stage_failed", stage_name="Model Pusher", error=str(custom_error))
            raise custom_error
        

    def run_pipeline(self) -> None:
        """
        Orchestrates the full flow with a safety gate at the evaluation stage.
        """
        try:
            self.log.info("entire_pipeline_run_initiated")
            # Start Ingestion, this captures the artifact which will eventually be passed to Data Validation
            data_ingestion = self.start_data_ingestion()
            data_validation = self.start_data_validation(data_ingestion_artifact=data_ingestion)
            data_transformation = self.start_data_transformation(data_ingestion_artifact=data_ingestion,
                                                                 data_validation_artifact=data_validation)
            model_trainer = self.start_model_trainer(data_transformation_artifact=data_transformation)
            model_evaluation = self.start_model_evaluation(data_ingestion_artifact=data_ingestion,
                                                           model_trainer_artifact=model_trainer)
            

            if model_evaluation.is_model_accepted:
                self.log.info("model_accepted_proceeding_to_push")
                self.start_model_pusher(model_evaluation_artifact=model_evaluation)
            else:
                self.log.warning("model_rejected_skipping_push", 
                                 reason="Trained model performance is not better than S3 model.")

            self.log.info("entire_pipeline_run_successful")
            
        except Exception as e:
            custom_error = e if isinstance(e, MyException) else MyException(e, sys)
            logger.error("entire_pipeline_run_failed", error=str(custom_error))
            raise custom_error

