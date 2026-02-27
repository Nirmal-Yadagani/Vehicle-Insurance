import sys
import pandas as pd
import numpy as np
from typing import Optional

from src.configuration.mongo_db_connection import MongoDBClient
from src.constants import DATABASE_NAME
from src.exception import MyException
from src.logger import logging

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
            raise MyException(e, sys)
        
        
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

        try:
            if database_name is None:
                collection = self.mongo_client.database[collection_name]
            else:
                collection = self.mongo_client[database_name][collection_name]
            
            logging.info(f"fetching data from collection {collection_name}")
            data = list(collection.find())
            logging.info(f"data fetched successfully from collection {collection_name} with {len(data)} records.")
            df = pd.DataFrame(data)

            if 'id' in df.columns.to_list():
                df.drop(columns=["id"], inplace=True)
            df.replace({"na":np.nan},inplace=True)
            return df
        
        except Exception as e:
            raise MyException(e, sys)