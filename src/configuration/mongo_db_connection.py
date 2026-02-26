import os
import sys
import pymongo
import certifi
from dotenv import load_dotenv


from src.logger import logging
from src.exception import MyException

from src.constants import DATABASE_NAME, MONGO_DB_URL_KEY

# Silence the PyMongo internal debug logs
logging.getLogger("pymongo").setLevel(logging.WARNING)
# If using motor (async MongoDB)
logging.getLogger("motor").setLevel(logging.WARNING)


ca = certifi.where()

class MongoDBClient:
    """MongoDBClient is a class that provides a connection to a MongoDB database. It uses the pymongo library to connect to the database and provides methods to interact with the database."""
    # class variable to hold the MongoDB client instance, shared across all instances of MongoDBClient to ensure only one connection is established
    client = None
    def __init__(self, database_name: str=DATABASE_NAME) -> None:
        """Initializes the MongoDB client and connects to the specified database. If the client is already initialized, it reuses the existing connection."""
        try:
            # check if the client is already initialized to avoid multiple connections
            if MongoDBClient.client is None:
                load_dotenv()
                self.mongo_db_url = os.getenv(MONGO_DB_URL_KEY)

                # if the MongoDB URL is not found in environment variables, raise an exception
                if not self.mongo_db_url:
                    raise MyException(f"MongoDB URL not found in environment variables with key: {MONGO_DB_URL_KEY}", sys)
                
                MongoDBClient.client = pymongo.MongoClient(self.mongo_db_url, tlsCAFile=ca)

            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name
            logging.info("MongoDB client initialized successfully.")

        except Exception as e:
            logging.error(f"Error initializing MongoDB client: {e}")
            raise MyException(e, sys)
        




