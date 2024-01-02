import time
import logging

# Configure logging to save results to a file
# logging.basicConfig(filename='function_timing.log', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

# Decorator function to measure the execution time of a function
def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        log_message = f"{func.__name__} took {execution_time:.4f} seconds to execute."
        logger.info(log_message)
        return result
    return wrapper