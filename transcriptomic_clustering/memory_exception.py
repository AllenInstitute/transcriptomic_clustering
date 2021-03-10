import sys
import traceback

def log_exception(exception: BaseException, expected: bool = True):
    """Prints the passed BaseException to the console, including traceback.

    :param exception: The BaseException to output.
    :param expected: Determines if BaseException was expected.
    """
    output = "[{}] {}: {}".format('EXPECTED' if expected else 'UNEXPECTED', type(exception).__name__, exception)
    print(output)
    exc_type, exc_value, exc_traceback = sys.exc_info()
    traceback.print_tb(exc_traceback)
