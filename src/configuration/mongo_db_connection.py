import os
import sys

import logging
import pymongo
import certifi
import structlog
from dotenv import load_dotenv

from src.exception import MyException
from src.constants import DATABASE_NAME, MONGO_DB_URL_KEY

# get the structlog logger
logger = structlog.get_logger()

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
        log = logger.bind(database_name=database_name)
        try:
            # check if the client is already initialized to avoid multiple connections
            if MongoDBClient.client is None:
                log.info("Initializing MongoDB client for the first time.")
                load_dotenv()
                self.mongo_db_url = os.getenv(MONGO_DB_URL_KEY)

                # if the MongoDB URL is not found in environment variables, raise an exception
                if not self.mongo_db_url:
                    raise MyException(f"MongoDB URL not found in environment variables with key: {MONGO_DB_URL_KEY}", sys)
                
                MongoDBClient.client = pymongo.MongoClient(self.mongo_db_url, tlsCAFile=ca)

            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name
            log.info("MongoDB client initialized successfully.")

        except Exception as e:
            custom_error = e if isinstance(e, MyException) else MyException(e, sys)
            log.error(f"Error initializing MongoDB client", error=str(custom_error))
            raise custom_error