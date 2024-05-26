import logging

log_levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


class LoggerCreator:
    @staticmethod
    def create_logger(log_path='./log', logging_name=None, level=logging.INFO, args=None):
        """create a logger
        Args:
            name (str): name of the logger
            level: level of logger

        Raises:
            ValueError is name is None
        """

        if logging_name is None:
            raise ValueError("name for logger cannot be None")

        logger = logging.getLogger(logging_name)
        logger.setLevel(level=logging.DEBUG)
        handler = logging.FileHandler(log_path, encoding='UTF-8', mode='w')
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.addHandler(console)
        return logger

