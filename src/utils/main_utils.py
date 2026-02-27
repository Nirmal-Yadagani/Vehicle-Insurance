import os
import sys

import numpy as np
import dill
import yaml

from src.exception import MyException
from src.logger import logging



def read_yaml_file(file_path: str) -> dict:
    """
    Reads a YAML file and returns its contents as a dictionary.

    Args:
        file_path (str): The system path to the YAML configuration file.

    Returns:
        dict: The parsed content of the YAML file.

    Raises:
        MyException: If the file is not found or contains invalid YAML syntax.
    """
    try:
        with open(file_path, 'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    
    except Exception as e:
        raise MyException(e, sys)
    

def write_yaml_file(file_path: str, content: object, replace: bool=False) -> None:
    """
    Writes a Python object (usually a dict) to a YAML file.

    Args:
        file_path (str): The target system path for the YAML file.
        content (object): The data to be serialized into YAML format.
        replace (bool): If True, deletes the existing file before writing a new one. 
                        Defaults to False.

    Returns:
        None

    Raises:
        MyException: If directory creation or file writing fails.
    """
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as yaml_file:
            yaml.safe_dump(content, file_path, )
        logging.info(f"Saved YAML file at {file_path}")

    except Exception as e:
        raise MyException(e, sys)
    

def load_object(file_path: str) -> object:
    """
    Loads a serialized object (model, scaler, etc.) from the specified file path.

    Args:
        file_path (str): The system path where the pickled object is stored.

    Returns:
        object: The deserialized Python object (e.g., a Machine Learning model).

    Raises:
        MyException: If the file is missing or corrupted during the loading process.
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise MyException(e, sys)
    

def save_object(obj: object, file_path: str) -> None:
    """
    Serializes a Python object and saves it to a specified file path using dill.

    Args:
        file_path (str): The system path where the object will be saved.
        obj (object): The Python object (model, transformer, etc.) to be serialized.

    Returns:
        None

    Raises:
        MyException: If any error occurs during directory creation or serialization.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"Saved the object to {file_path}")

    except Exception as e:
        raise MyException(e, sys)
    

def save_numpy_array_data(arr: np.array, file_path: str) -> None:
    """
    Saves a numpy array data to a specified file path.

    Args:
        array (np.array): The numpy array data to be saved.
        file_path (str): The system path where the numpy array will be saved.

    Returns:
        None

    Raises:
        MyException: If any error occurs during directory creation or file writing.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, arr)
        logging.info(f"Saved the Numpy array at {file_path}")

    except Exception as e:
        raise MyException(e, sys)


def load_numpy_array_data(file_path: str) -> np.array:
    """
    Loads a numpy array from a specified file path.

    Args:
        file_path (str): The system path where the numpy array (.npy) is stored.

    Returns:
        np.array: The loaded numpy array data.

    Raises:
        MyException: If the file cannot be found or is not a valid numpy binary file.
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)

    except Exception as e:
        raise MyException(e, sys)