import os
import sys

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, TargetEncoder, OneHotEncoder

from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationAritfact
from src.entity.config_entity import DataTransformationConfig
from src.exception import MyException
from src.logger import logger
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
            self.log = logger.bind(transformation_dir=self.data_transformation_config.data_tranformation_dir)
        
        except Exception as e:
            custom_error = e if isinstance(e, MyException) else MyException(e, sys)
            self.log.error("Error initializing DataTransformation", error=str(custom_error))
            raise custom_error


    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        
        except Exception as e:
            if isinstance(e, MyException):
                raise e
            raise MyException(e, sys)
        
    
    def get_data_transformer_object(self) -> ColumnTransformer:
        try:
            self.log.info('Creating column transformer...')
            target_encoder_transformer = ('Target_encoder', TargetEncoder(), self._schema_config['target_encoder_columns'])
            std_scaler_transformer = ('std_scaler', StandardScaler(), self._schema_config['std_scaler_columns'])
            mm_scaler_transformer = ('minmax_scaler', MinMaxScaler(), self._schema_config['mm_columns'])
            onehot_encoder_transformer = ('Onehot_encoder', OneHotEncoder(drop='first', sparse_output=False), self._schema_config['onehot_encoder_columns'])

            transformers = [target_encoder_transformer,
                            std_scaler_transformer,
                            mm_scaler_transformer,
                            onehot_encoder_transformer]
            
            ct = ColumnTransformer(transformers=transformers, remainder='drop').set_output(transform='pandas')
            self.log.info("Column transformer created successfully.", transformer_count=len(transformers))
            return ct
        
        except Exception as e:
            custom_error = e if isinstance(e, MyException) else MyException(e, sys)
            self.log.error("Error occurred while creating column transformer.", error=str(custom_error))
            raise custom_error         


    def initiate_data_transformation(self) -> DataTransformationAritfact:
        try:
            self.log.info('Data transformation started...')
            if not self.data_validation_artifact.validation_status:
                self.log.error("Data validation failed. Cannot proceed with data transformation.", message=self.data_validation_artifact.message)
                raise Exception(self.data_validation_artifact.message)
            
            train_df = DataTransformation.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df = DataTransformation.read_data(self.data_ingestion_artifact.test_file_path)
            self.log.info('Raw data loaded', train_shape=train_df.shape, test_shape=test_df.shape)

            input_features_train_df = train_df.drop(columns=[TARGET_COLUMN])
            target_features_train_df = train_df[TARGET_COLUMN]

            input_features_test_df = test_df.drop(columns=[TARGET_COLUMN])
            target_features_test_df = test_df[TARGET_COLUMN]

            ct = self.get_data_transformer_object()

            self.log.info('Applying fit_transform to training data...')
            input_features_train_transformed = ct.fit_transform(input_features_train_df, target_features_train_df)

            self.log.info('Applying transform to testing data...')
            input_features_test_transformed = ct.transform(input_features_test_df)
            
            smoteen = SMOTEENN(sampling_strategy="minority")
            input_features_train_final, target_features_train_final = smoteen.fit_resample(input_features_train_transformed, target_features_train_df)
            self.log.info("Smoteenn applied on train data.",after_smote_rows=len(input_features_train_final))

            train_arr = np.c_[input_features_train_final, target_features_train_final]
            test_arr = np.c_[input_features_test_transformed.to_numpy(), target_features_test_df.to_numpy()]

            dir_name = self.data_transformation_config.data_tranformation_dir
            os.makedirs(dir_name, exist_ok=True)

            save_object(ct, self.data_transformation_config.transformed_object_file_path)
            save_numpy_array_data(train_arr, self.data_transformation_config.transformed_train_file_path)
            save_numpy_array_data(test_arr, self.data_transformation_config.transformed_test_file_path)
            self.log.info('Transformation artifact saved successfully.')

            return DataTransformationAritfact(self.data_transformation_config.transformed_object_file_path,
                                              self.data_transformation_config.transformed_train_file_path,
                                              self.data_transformation_config.transformed_test_file_path)

        except Exception as e:
            custom_error = e if isinstance(e, MyException) else MyException(e, sys)
            self.log.error("Error during data transformation process.", error=str(custom_error))
            raise custom_error

       