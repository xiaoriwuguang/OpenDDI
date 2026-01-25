import sys
import os
import logging
import pprint
import time
from utils.config import LOG_PATH
class Logger(object):
    """
    Custom logger class for handling both file and console logging.
    
    Args:
        filename (str): Path to the log file
    """
    
    def __init__(self, filename):
        dir_name = os.path.dirname(filename)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        self.logger = logging.getLogger(filename)
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s.%(msecs)03d: %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')

        # write into file
        fileHandler = logging.FileHandler(filename)
        fileHandler.setLevel(logging.DEBUG)
        fileHandler.setFormatter(formatter)

        # show on console
        consoleHandler = logging.StreamHandler(sys.stdout)
        consoleHandler.setLevel(logging.DEBUG)
        consoleHandler.setFormatter(formatter)

        # add to Handler
        self.logger.addHandler(fileHandler)
        self.logger.addHandler(consoleHandler)

    def _flush(self):
        """
        Flush all logger handlers to ensure all messages are written.
        """
        for handler in self.logger.handlers:
            handler.flush()

    def debug(self, message):
        """
        Log debug level message.
        
        Args:
            message (str): Message to log
        """
        self.logger.debug(message)
        self._flush()

    def info(self, message):
        """
        Log info level message.
        
        Args:
            message (str): Message to log
        """
        self.logger.info(message)
        self._flush()

    def warning(self, message):
        """
        Log warning level message.
        
        Args:
            message (str): Message to log
        """
        self.logger.warning(message)
        self._flush()

    def error(self, message):
        """
        Log error level message.
        
        Args:
            message (str): Message to log
        """
        self.logger.error(message)
        self._flush()

    def critical(self, message):
        """
        Log critical level message.
        
        Args:
            message (str): Message to log
        """
        self.logger.critical(message)
        self._flush()

def create_logger(args):
    """
    Create and configure a logger instance with timestamp and model name.
    
    Args:
        args: Configuration arguments containing model information
        
    Returns:
        Logger: Configured logger instance
    """
    timestamp = time.time()
    current_time_str = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(timestamp))
    model_name = args.model
    run_id = f"{timestamp:.8f}"
    log_dir = os.path.join(LOG_PATH , "log" , model_name)
    logger_name = os.path.join(log_dir, current_time_str + ".log")
    logger = Logger(logger_name)

    logger.info(f"my pid: {os.getpid()}")
    formatted_args = pprint.pformat(args)
    logger.info(formatted_args)

    return logger
