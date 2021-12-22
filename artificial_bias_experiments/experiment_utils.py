import logging
import os
import sys
import time
from typing import List, Optional

import pandas as pd

TargetAttr = str


def reorder_columns(df: pd.DataFrame, target_column: TargetAttr) -> pd.DataFrame:
    """
    Generates a dafaframe with reordered columns, such that the given target column is the last colum
    :param df:
    :param target_column:
    :return:
    """
    if target_column not in df.columns:
        message = f"the given target column {target_column} is not a column of the given dataframe"
        raise Exception(message)
    columns = df.columns
    reordered_columns = []
    for possibly_other_column in columns:
        if str(possibly_other_column) != str(target_column):
            reordered_columns.append(possibly_other_column)
    # reordered_columns = [other_col for other_col in columns if str(other_col) is not str(target_column)]
    reordered_columns.append(target_column)
    new_df = df[reordered_columns]
    return new_df


def create_logger(logger_name: str, log_file_name: str) -> logging.Logger:
    # create formatters
    file_log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # create a logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # create handlers
    file_handler = logging.FileHandler(log_file_name, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_log_formatter)

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_log_formatter)

    # add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def create_stdout_logger(logger_name: str = None) -> logging.Logger:

    if logger_name is None:
        logger_name = "stdout_logger"

    # create formatters
    console_log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # create a logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # create handlers
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_log_formatter)

    # add the handlers to the logger
    logger.addHandler(console_handler)
    return logger


def close_logger(logger: logging.Logger) -> None:
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)


def get_header_attributes(abs_file_name: str) -> List[str]:
    header_attributes: List[str]
    with open(abs_file_name, 'r') as input_data_file:
        header_line: str = input_data_file.readline()
        header_line_without_line_break: str = header_line.replace('\n', '')
        header_attributes = header_line_without_line_break.split(',')
    return header_attributes


def print_or_log(message: str, logger: Optional[logging.Logger], verbose: bool) -> None:
    if logger is not None:
        logger.info(message)
    elif verbose:
        print(message)


def check_file_last_modified_duration_in_hours(abs_file_name: str) -> Optional[float]:
    if os.path.isfile(abs_file_name):
        time_point_last_modified_s: float = os.path.getmtime(abs_file_name)
        current_time_s: float = time.time()
        time_range_since_last_modified_s = current_time_s - time_point_last_modified_s
        minutes = time_range_since_last_modified_s / 60.0  # 120 minutes
        hours = minutes / 60  # 2 hours
        return hours
    else:
        return None


def file_does_not_exist_or_has_been_created_earlier_than_(abs_file_name, nb_hours):
    if not os.path.isfile(abs_file_name):
        return True  # file does not exists
    else:
        hours_since_last_modified: float = check_file_last_modified_duration_in_hours(
            abs_file_name
        )
        if hours_since_last_modified is None:
            raise Exception(f"Sudden disappearance of file {abs_file_name}")
        elif hours_since_last_modified > nb_hours:
            return True  # file is old enough
    return False  # File exists and but is not old enough

