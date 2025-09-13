import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def logging_message(func):
    def wrapper(*args, **kwargs):
        self = args[0]
        name = self.__class__.__name__
        message = func(*args, **kwargs)
        logger.info(f"\n\n============ {name} Response ============\n{message}\n\n")
        return message

    return wrapper


def logging_user_input(name: str, message: str):
    logger.info(f"\n\n============ {name} Response ============\n{message}\n\n")
