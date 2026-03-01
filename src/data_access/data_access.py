import sys

import pandas as pd
import numpy as np
from typing import Optional

from src.configuration.mongo_db_connection import MongoDBClient
from src.constants import DATABASE_NAME
from src.exception import MyException
from src.logger import logger

class DataAccess:
    """DataAccess class is responsible for fetching data from MongoDB collections and converting it into pandas DataFrames for further processing in the machine learning pipeline."""

    def __init__(self) -> None:
        """Initializes the DataAccess class and establishes a connection to the MongoDB database.
        Raises:
            MyException: If there is an error while initializing the MongoDB client.
        """
        try:
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)
        except Exception as e:
            custom_error = e if isinstance(e, MyException) else MyException(e, sys)
            logger.error('Error occurred while initializing MongoDB client.', error=str(custom_error))
            raise custom_error
        
        
    def export_collection_as_dataframe(self, collection_name: str, database_name: Optional[str] = None) -> pd.DataFrame:
        """Fetches data from the specified MongoDB collection and returns it as a pandas DataFrame.
        Args:
            collection_name (str): The name of the MongoDB collection to fetch data from.
            database_name (Optional[str]): The name of the MongoDB database. If not provided, it will use the default database specified in the MongoDB client.
        Returns:
            pd.DataFrame: A DataFrame containing the data fetched from the MongoDB collection.
        Raises:
            MyException: If there is an error while fetching data from the MongoDB collection or converting it to a DataFrame.
        """
        log = logger.bind(collection_name=collection_name, database_name=database_name or DATABASE_NAME)
        try:
            if database_name is None:
                collection = self.mongo_client.database[collection_name]
            else:
                collection = self.mongo_client[database_name][collection_name]
            
            log.info(f"fetching data from mongoDB.")
            data = list(collection.find())
            log.info(f"data fetched successfully.", record_count=len(data))
            df = pd.DataFrame(data)

            if 'id' in df.columns.to_list():
                df.drop(columns=["id"], inplace=True)
                log.info('Dropped "id" column from DataFrame.', remaining_columns=df.columns.to_list())

            df.replace({"na":np.nan},inplace=True)
            log.info("dataframe transformed successfully.", dataframe_shape=df.shape)
            return df
        
        except Exception as e:
            custom_error = e if isinstance(e, MyException) else MyException(e, sys)
            log.error('Error occurred while exporting data from MongoDB collection.', error=str(custom_error))
            raise custom_error