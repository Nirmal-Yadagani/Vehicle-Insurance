import sys

from pandas import DataFrame
from sklearn.compose import ColumnTransformer

from src.exception import MyException
from src.logger import logging


class MyModel:
    def __init__(self, preprocessing_object: ColumnTransformer, trained_model_object: object):
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object


    def predict(self, dataframe: DataFrame) -> DataFrame:
        try:
            logging.info('Started prediction process.')

            transformed_features = self.preprocessing_object.transform(dataframe)

            logging.info('Using the trained model to get predictions')
            predictions = self.trained_model_object.predict(transformed_features)

            return predictions
        
        except Exception as e:
            raise MyException(e, sys)
        

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}"
    

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}"
    