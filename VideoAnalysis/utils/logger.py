import logging
from rich.logging import RichHandler

def get_pylogger(name=__name__):
    """Get a logger with a custom format."""
    
    logging.basicConfig(
        level="INFO",
        format="%(message)s",
        datefmt="[%m/%d %H:%M:%S]",
        handlers=[RichHandler()]
    )
    root_logger = logging.getLogger(name)
    
    return root_logger
