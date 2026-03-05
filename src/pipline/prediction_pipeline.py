import sys

from pandas import DataFrame

from src.entity.config_entity import VehiclePredictorConfig
from src.entity.s3_estimator import S3Estimator
from src.exception import MyException
from src.logger import logger



class VehicleData:
    """
    Acts as a Data Transfer Object (DTO) to structure user input into a format 
    suitable for the machine learning model.
    """
    def __init__(self, 
                 Gender,
                 Age,
                 Driving_License,
                 Region_Code,
                 Previously_Insured,
                 Annual_Premium,
                 Policy_Sales_Channel,
                 Vintage,
                 Vehicle_Age,
                 Vehicle_Damage
                 ):
        """
        Initializes the VehicleData object with raw user attributes.

        Args:
            Gender (str): Gender of the user.
            Age (int): Age of the user.
            Driving_License (int): License status (0 or 1).
            Region_Code (int): Geographic region code.
            Previously_Insured (int): Insurance status (0 or 1).
            Annual_Premium (float): Yearly premium amount.
            Policy_Sales_Channel (float): Channel through which policy was sold.
            Vintage (float): Number of days associated with the company.
            Vehicle_Age (str): Age of the vehicle.
            Vehicle_Damage (str): Damage history of the vehicle.
        """
        try:
            self.Gender = Gender
            self.Age = Age
            self.Driving_License = Driving_License
            self.Region_Code = Region_Code
            self.Previously_Insured = Previously_Insured
            self.Annual_Premium = Annual_Premium
            self.Policy_Sales_Channel = Policy_Sales_Channel
            self.Vintage = Vintage
            self.Vehicle_Age = Vehicle_Age
            self.Vehicle_Damage = Vehicle_Damage

        except Exception as e:
            raise MyException(e, sys) from e
        
    
    def get_vehicle_input_data_frame(self) -> DataFrame:
        """
        Converts the class attributes into a single-row Pandas DataFrame.

        Returns:
            DataFrame: A pandas DataFrame containing the mapped input features.

        Raises:
            MyException: If an error occurs during dictionary-to-dataframe conversion.
        """
        try:
            input_data = {
                "Gender": [self.Gender],
                "Age": [self.Age],
                "Driving_License": [self.Driving_License],
                "Region_Code": [self.Region_Code],
                "Previously_Insured": [self.Previously_Insured],
                "Annual_Premium": [self.Annual_Premium],
                "Policy_Sales_Channel": [self.Policy_Sales_Channel],
                "Vintage": [self.Vintage],
                "Vehicle_Age": [self.Vehicle_Age],
                "Vehicle_Damage": [self.Vehicle_Damage]
            }

            return DataFrame(input_data)

        except Exception as e:
            raise MyException(e, sys) from e
        


class InsuranceSubscriptionClassifier:
    """
    Handles the high-level orchestration of the prediction process, 
    including model retrieval from S3 and inference execution.
    """
    def __init__(self, prediction_pipeline_config: VehiclePredictorConfig=VehiclePredictorConfig) -> None:
        """
        Initializes the classifier with necessary S3 bucket and model path configurations.

        Args:
            prediction_pipeline_config (VehiclePredictorConfig): Configuration for model storage locations.
        """
        try:
            self.prediction_pipeline_config = prediction_pipeline_config

        except Exception as e:
            raise MyException(e, sys) from e
        
    def predict(self, dataframe) -> str:
        """
        Uses the S3-hosted model to predict insurance subscription likelihood.

        This method triggers the S3Estimator to load the model (if not already cached) 
        and performs inference on the provided dataframe.

        Args:
            dataframe (DataFrame): The input features prepared via get_vehicle_input_data_frame.

        Returns:
            str: "Yes" if the user is likely to subscribe, "No" otherwise.

        Raises:
            MyException: If an error occurs during model retrieval or prediction.
        """
        try:
            self.s3 = S3Estimator(self.prediction_pipeline_config.model_bucket_name,
                                  self.prediction_pipeline_config.model_file_path)

            prediction = self.s3.predict(dataframe=dataframe)

            return "Yes" if prediction else "No"
        
        except Exception as e:
            raise MyException(e, sys) from e