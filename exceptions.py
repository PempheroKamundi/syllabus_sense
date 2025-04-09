class SyllabusSenseException(Exception):
    """Base exception class for all SyllabusSense application exceptions.

    This serves as the parent class for all custom exceptions in the SyllabusSense
    system, allowing for consistent error handling and categorization.
    """

    def __init__(self, message="An error occurred in the SyllabusSense system"):
        self.message = message
        super().__init__(self.message)


class InvalidOutputDirectoryError(SyllabusSenseException):
    """Exception raised when the provided output path is not a valid directory."""

    def __init__(self, path, message="Provided path is not a valid directory. Please provide a valid path, not a file"):
        self.path = path
        self.message = f"{message}: {path}"
        super().__init__(self.message)