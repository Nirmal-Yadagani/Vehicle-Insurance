import sys

from pandas import DataFrame
from sklearn.compose import ColumnTransformer

from src.exception import MyException
from src.logger import logger


class MyModel:
    """
    A production wrapper that bundles the preprocessing pipeline and the trained model.
    Ensures that raw data is consistently transformed before making predictions.
    """
    def __init__(self, preprocessing_object: ColumnTransformer, trained_model_object: object):
        """
        Initializes the model wrapper.

        Args:
            preprocessing_object (ColumnTransformer): The fitted transformer (scalers, encoders).
            trained_model_object (object): The fitted estimator (e.g., RandomForest).
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object
        self.log = logger.bind(model_type=type(self.trained_model_object).__name__)


    def predict(self, dataframe: DataFrame) -> DataFrame:
        """
        Transforms input data and generates predictions.

        Args:
            dataframe (DataFrame): Raw input data.

        Returns:
            DataFrame: Model predictions.
        
        Raises:
            MyException: If transformation or prediction fails.
        """
        try:
            self.log.info('prediction_started', input_rows=dataframe.shape[0], input_cols=dataframe.shape[1])
            transformed_features = self.preprocessing_object.transform(dataframe)

            self.log.info('transformation_applied', transformed_shape=transformed_features.shape)            
            predictions = self.trained_model_object.predict(transformed_features.values)

            self.log.info('prediction_successful', count=len(predictions))
            return predictions
        
        except Exception as e:
            custom_error = e if isinstance(e, MyException) else MyException(e, sys)
            self.log.error('prediction_failed', error=str(custom_error))            
            raise custom_error
        

    def __repr__(self):
        return f"MyModel(trained_model={type(self.trained_model_object).__name__})"

    def __str__(self):
        return f"Model Wrapper for {type(self.trained_model_object).__name__}"
    