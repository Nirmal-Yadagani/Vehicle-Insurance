import sys
import os

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, TargetEncoder, OneHotEncoder

from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationAritfact
from src.entity.config_entity import DataTransformationConfig
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file


class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        
        except Exception as e:
            raise MyException(e, sys)


    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        
        except Exception as e:
            raise MyException(e, sys)
        
    
    def get_data_transformer_object(self) -> object:
        logging.info("Entered get_data_transformer_object method of DataTransformation class")
        try:
            logging.info('Creating transformers for coloumtransformer of sklearn')
            target_encoder_transformer = ('Target_encoder', TargetEncoder(), self._schema_config['target_encoder_columns'])
            logging.info('Created Target encoder')
            std_scaler_transformer = ('std_scaler', StandardScaler(), self._schema_config['std_scaler_columns'])
            logging.info('Created standard scaler')
            mm_scaler_transformer = ('minmax_scaler', MinMaxScaler(), self._schema_config['mm_columns'])
            logging.info('Created minmax scaler')
            onehot_encoder_transformer = ('Onehot_encoder', OneHotEncoder(drop='first', sparse_output=False), self._schema_config['onehot_encoder_columns'])
            logging.info('Created onehot encoder')

            transformers = [target_encoder_transformer,
                            std_scaler_transformer,
                            mm_scaler_transformer,
                            onehot_encoder_transformer]
            
            ct = ColumnTransformer(transformers=transformers, remainder='drop').set_output(transform='pandas')
            logging.info('Created colountransformer object successfully')
            logging.info('Exited get_data_transformer_object method of DataTransformation class')
            return ct
        
        except Exception as e:
            logging.exception("Exception occurred in get_data_transformer_object method of DataTransformation class")
            raise MyException(e, sys)         


    def initiate_data_transformation(self) -> DataTransformationAritfact:
        try:
            logging.info('Data transformation started')
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)
            
            train_df = DataTransformation.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df = DataTransformation.read_data(self.data_ingestion_artifact.test_file_path)
            logging.info('Train-Test data loaded')

            input_features_train_df = train_df.drop(columns=[TARGET_COLUMN])
            target_features_train_df = train_df[TARGET_COLUMN]

            input_features_test_df = test_df.drop(columns=[TARGET_COLUMN])
            target_features_test_df = test_df[TARGET_COLUMN]
            logging.info("Input and Target cols defined for both train df and test df")

            logging.info('Starting data transformation')
            ct = self.get_data_transformer_object()
            logging.info('Got the column transformer object')

            smoteen = SMOTEENN(sampling_strategy="minority")
            steps = [('ct', ct),
                     ('smoteen', smoteen)]
            
            preprocessor = Pipeline(steps=steps).set_output(transform='pandas')
            logging.info('Created the preprocessor object with smoteen included')

            input_features_train_final, target_features_train_final = preprocessor.fit_resample(input_features_train_df, target_features_train_df)

            train_arr = np.c_[input_features_train_final, target_features_train_final]
            test_arr = np.c_[input_features_test_df, target_features_test_df]
            logging.info("feature-target concatenation done for train-test df")


            dir_name = self.data_transformation_config.data_tranformation_dir
            os.makedirs(dir_name, exist_ok=True)
            save_object(preprocessor, self.data_transformation_config.transformed_object_file_path)
            save_numpy_array_data(train_arr, self.data_transformation_config.transformed_train_file_path)
            save_numpy_array_data(test_arr, self.data_transformation_config.transformed_test_file_path)
            logging.info("Saved transform object and transformed files.")

            logging.info("Data transformation successful.")

            return DataTransformationAritfact(self.data_transformation_config.transformed_object_file_path,
                                              self.data_transformation_config.transformed_train_file_path,
                                              self.data_transformation_config.transformed_test_file_path)

        except Exception as e:
            raise MyException(e ,sys)

       