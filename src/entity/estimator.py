import sys

from pandas import DataFrame
from sklearn.compose import ColumnTransformer

from src.exception import MyException
from src.logger import logger


class MyModel:
    def __init__(self, preprocessing_object: ColumnTransformer, trained_model_object: object):
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object
        self.log = logger.bind(model_type=type(self.trained_model_object).__name__)

    def predict(self, dataframe: DataFrame) -> DataFrame:
        try:
            self.log.info('Started prediction process.', input_shape=dataframe.shape)

            transformed_features = self.preprocessing_object.transform(dataframe)

            self.log.info('Using the trained model to get predictions', transformed_shape=transformed_features.shape)
            predictions = self.trained_model_object.predict(transformed_features)

            self.log.info('Prediction process completed successfully.', predictions_shape=len(predictions))

            return predictions
        
        except Exception as e:
            custom_error = e if isinstance(e, MyException) else MyException(e, sys)
            self.log.error('Error occurred during prediction process.', error=str(custom_error))
            raise custom_error
        

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}"
    

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}"
    