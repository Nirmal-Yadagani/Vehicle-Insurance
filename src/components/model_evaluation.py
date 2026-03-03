
import sys
from typing import Optional
from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import f1_score

from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import DataIngestionArtifact, ModelTrainerArtifact, ModelEvaluationArtifact
from src.logger import logger
from src.exception import MyException
from src.entity.s3_estimator import S3Estimator
from src.utils.main_utils import load_object
from src.constants import TARGET_COLUMN


@dataclass
class EvaluateModelResponse:
    """
    Data structure to hold the results of the model comparison.
    """ 
    trained_model_f1_score: float
    best_model_f1_score: float
    is_model_accepted: bool
    difference: float



class ModelEvaluation:
    """
    Evaluates the newly trained model against the current best model in S3.
    Decides if the new model should replace the existing production model.
    """
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact,
                 model_eval_config: ModelEvaluationConfig):
        """
        Initializes the ModelEvaluation component.

        Args:
            data_ingestion_artifact: Artifact containing paths to test data.
            model_trainer_artifact: Artifact containing the newly trained model and its metrics.
            model_eval_config: Configuration for S3 bucket and model paths.
        """
        try:
            self.model_eval_config = model_eval_config
            self.model_trainer_artifact = model_trainer_artifact
            self.data_ingestion_artifact = data_ingestion_artifact
            self.log = logger.bind(stage="model evaluation")

        except Exception as e:
            if isinstance(e, MyException):
                raise e
            raise MyException(e, sys)


    def get_best_model(self) -> Optional[S3Estimator]:
        """
        Attempts to fetch the current production model from S3.

        Returns:
            Optional[S3Estimator]: The model wrapper if it exists in S3, else None.
        """
        try:
            bucket_name = self.model_eval_config.bucket_name
            model_path = self.model_eval_config.s3_model_key_path

            self.log.info("fetching best_model_from_s3", bucket=bucket_name, path=model_path)
            s3_estimator = S3Estimator(bucker_name=bucket_name, model_path=model_path)

            if s3_estimator.is_model_present(model_path=model_path):
                return s3_estimator
            
            self.log.info("no existing model found in s3")
            return None
        
        except Exception as e:
            custom_error = e if isinstance(e, MyException) else MyException(e, sys)
            self.log.error("error_fetching_best_model", error=str(custom_error))
            raise custom_error

    
    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Compares the performance of the trained model vs the S3 model using test data.

        Returns:
            EvaluateModelResponse: Metrics and decision on model acceptance.
        """
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            X, y = test_df.drop(columns=[TARGET_COLUMN]), test_df[TARGET_COLUMN]

            trained_model_f1_score = self.model_trainer_artifact.metric_artifact.f1_score

            best_model_f1_score = 0.0
            best_model = self.get_best_model()

            if best_model:
                self.log.info("evaluating s3_model on test_data")
                y_hat_best_model = best_model.predict(X)
                best_model_f1_score = f1_score(y, y_hat_best_model)

            difference = trained_model_f1_score - best_model_f1_score
            is_accepted = difference > 0  # Only accept if it's strictly better

            result = EvaluateModelResponse(
                trained_model_f1_score=trained_model_f1_score,
                best_model_f1_score=best_model_f1_score,
                is_model_accepted=is_accepted,
                difference=difference
            )

            self.log.info("model_comparison_result", 
                          trained_score=round(trained_model_f1_score, 4),
                          best_s3_score=round(best_model_f1_score, 4),
                          accepted=is_accepted,
                          improvement=round(difference, 4))

            return result
        
        except Exception as e:
            custom_error = e if isinstance(e, MyException) else MyException(e, sys)
            self.log.error("model_evaluation_failed", error=str(custom_error))
            raise custom_error
        

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Orchestrates the evaluation process and returns an artifact.

        Returns:
            ModelEvaluationArtifact: Evaluation summary and paths.
        """
        self.log.info("initiate_model_evaluation_started")
        try:
            evaluate_model_response = self.evaluate_model()

            s3_model_path = self.model_eval_config.s3_model_key_path


            model_eval_artifact = ModelEvaluationArtifact(evaluate_model_response.is_model_accepted,
                                                          changed_accuracy=evaluate_model_response.difference,
                                                          trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                                                          s3_model_path=s3_model_path)
            self.log.info("model_evaluation_completed", 
                          accepted=model_eval_artifact.is_model_accepted)
            
            return model_eval_artifact
        
        except Exception as e:
            custom_error = e if isinstance(e, MyException) else MyException(e, sys)
            self.log.error("model_evaluation_orchestration_failed", error=str(custom_error))
            raise custom_error

