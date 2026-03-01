from typing import Tuple
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from src.exception import MyException
from src.logger import logger
from src.entity.artifact_entity import DataTransformationAritfact, ModelTrainerArtifact, ClassificationMetricArtifact
from src.entity.config_entity import ModelTrainerConfig
from src.entity.estimator import MyModel
from src.utils.main_utils import load_numpy_array_data, load_object, save_object


class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationAritfact,
                 model_trainer_config: ModelTrainerConfig):
        
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
            self.log = logger.bind(model_name="RandomForestClassifier")

        except Exception as e:
            custom_error = e if isinstance(e, MyException) else MyException(e, sys)
            logger.error("Error initializing ModelTrainer", error=str(custom_error))
            raise custom_error
        
    
    def get_model_object_and_report(self, train: np.array, test: np.array) -> Tuple[object, ClassificationMetricArtifact]:
        try:
            parameters = {
                "n_estimators": self.model_trainer_config._n_estimators,
                "criterion": self.model_trainer_config._criterion,
                "max_depth": self.model_trainer_config._max_depth,
                "min_samples_leaf": self.model_trainer_config._min_samples_leaf,
                "min_samples_split": self.model_trainer_config._min_samples_split,
                "random_state": self.model_trainer_config._random_state
            }

            self.log.info('Traiining started for RandomForestClassifier', hyperparameters=parameters)

            X_train, y_train = train[:,:-1], train[:,-1]
            X_test, y_test = test[:,:-1], test[:,-1]

            model = RandomForestClassifier(**parameters)
            
            model.fit(X_train, y_train)
            self.log.info('Model fitted successfully.')

            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

            self.log.info('evaluation metrics',accuracy=round(accuracy, 4), f1_score=round(f1, 4), precision=round(precision, 4), recall=round(recall, 4))

            metric_artifact = ClassificationMetricArtifact(f1_score=f1,
                                                           precision_score=precision,
                                                           recall_score=recall)
            
            return model, metric_artifact
        
        except Exception as e:
            custom_error = e if isinstance(e, MyException) else MyException(e, sys)
            logger.error("Error in get_model_object_and_report", error=str(custom_error))
            raise custom_error
        

    
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        self.log.info('Entered the initiate_model_trainer method of ModelTrainer class')
        try:
            train_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)

            trained_model, metric_artifact = self.get_model_object_and_report(train=train_arr, test=test_arr)

            preprocessing_object = load_object(self.data_transformation_artifact.transformed_object_file_path)

            current_accuracy = accuracy_score(test_arr[:, -1], trained_model.predict(test_arr[:, :-1]))
            expected_accuracy = self.model_trainer_config.expected_accuracy

            if current_accuracy < expected_accuracy:
                self.log.info('model performance is below thresold.', threshold=expected_accuracy, actual=current_accuracy)
                raise Exception(f"Model accuracy {current_accuracy} is lower than threshold {expected_accuracy}")
            
            self.log.info("Saving Production model",path=self.model_trainer_config.trained_model_file_path)
            my_model = MyModel(preprocessing_object=preprocessing_object,
                               trained_model_object=trained_model)
            
            save_object(my_model, self.model_trainer_config.trained_model_file_path)

            model_trainer_artifact = ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                                                          metric_artifact=metric_artifact)
            self.log.info(f'Model trainer completed', artifact=model_trainer_artifact.__dict__)
            return model_trainer_artifact

        except Exception as e:
            custom_error = e if isinstance(e, MyException) else MyException(e, sys)
            logger.error("Error in initiate_model_trainer", error=str(custom_error))
            raise custom_error
