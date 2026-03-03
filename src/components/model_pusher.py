import sys

from src.cloud_storage.aws_storage import SimpleStorageService
from src.exception import MyException
from src.logger import logger
from src.entity.config_entity import ModelPusherConfig
from src.entity.artifact_entity import ModelEvaluationArtifact, ModelPusherArtifact
from src.entity.s3_estimator import S3Estimator



class ModelPusher:
    """
    Handles the deployment of the validated model to AWS S3.
    This component is responsible for pushing the 'Champion' model to the 
    production bucket so that the prediction service can consume it.
    """
    def __init__(self, model_evaluation_artifact: ModelEvaluationArtifact,
                 model_pusher_config: ModelPusherConfig):
        """
        Initializes the ModelPusher with evaluation results and S3 configurations.

        Args:
            model_evaluation_artifact (ModelEvaluationArtifact): Results from the evaluation stage.
            model_pusher_config (ModelPusherConfig): Configuration for S3 bucket and key paths.
        """
        try:
            self.s3 = SimpleStorageService()
            self.model_eval_artifact = model_evaluation_artifact
            self.model_pusher_config = model_pusher_config

            # Contextual logger for the pusher
            self.log = logger.bind(
                bucket=model_pusher_config.bucket_name,
                s3_path=model_pusher_config.s3_model_key_path
            )

            self.s3_estimator = S3Estimator(bucker_name=model_pusher_config.bucket_name,
                                            model_path=model_pusher_config.s3_model_key_path)

        except Exception as e:
            custom_error = e if isinstance(e, MyException) else MyException(e, sys)
            logger.error("model_pusher_init_failed", error=str(custom_error))
            raise custom_error
        

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Triggers the upload of the local trained model to S3.

        Returns:
            ModelPusherArtifact: Metadata containing the destination bucket and S3 path.
        
        Raises:
            MyException: If the upload process fails.
        """
        self.log.info("model_pusher_process_initiated")
        try:
            self.log.info("uploading_model_to_s3", 
                          local_source=self.model_eval_artifact.trained_model_path)
            self.s3_estimator.save_model(self.model_eval_artifact.trained_model_path)
            self.log.info("model_successfully_pushed_to_production")

            model_pusher_artifact = ModelPusherArtifact(self.model_pusher_config.bucket_name, self.model_pusher_config.s3_model_key_path)

            return model_pusher_artifact
        
        except Exception as e:
            custom_error = e if isinstance(e, MyException) else MyException(e, sys)
            self.log.error("model_push_failed", error=str(custom_error))
            raise custom_error
