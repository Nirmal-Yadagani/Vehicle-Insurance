import sys
import logging


def error_message_detail(error: Exception, error_detail: sys) -> str:
    """Returns the error message with details about the file name and line number where the error occurred.
    Args:
        error (Exception): The exception object that was raised.
        error_detail (sys): The sys module that contains information about the current state of the Python interpreter.
    Returns:
        str: A string containing the error message with details about the file name and line number where the error occurred.
    """
    _, _, exc_tb = error_detail.exc_info()

    file_name = exc_tb.tb_frame.f_code.co_filename

    line_number = exc_tb.tb_lineno

    error_message = f"Error occurred in script: {file_name} at line number: {line_number} with error message: {str(error)}"

    logging.error(error_message)

    return error_message


class MyException(Exception):
    """Custom exception class that inherits from the built-in Exception class.
    Args:
        error_message (str): The error message to be associated with the exception.
        error_detail (sys): The sys module that contains information about the current state of the Python interpreter.
    """
    def __init__(self, error_message: str, error_detail: sys):
        """Initializes the MyException object with the error message and details about the error.
        Args:
            error_message (str): The error message to be associated with the exception.
            error_detail (sys): The sys module that contains information about the current state of the Python interpreter.
        """
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self) -> str:
        """Returns the string representation of the exception."""
        return self.error_message