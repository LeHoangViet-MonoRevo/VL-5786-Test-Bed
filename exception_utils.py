class ModelPredictionException(Exception):
    def __init__(self, message, errors):
        super().__init__(message)
        self.errors = errors


class SearchException(Exception):
    def __init__(self, message, errors):
        super().__init__(message)
        self.errors = errors


class ValidationException(Exception):
    def __init__(self, message, errors):
        super().__init__(message)
        self.errors = errors


class ImageParserException(Exception):
    def __init__(self, message, errors):
        super().__init__(message)
        self.errors = errors


class AISystemError(Exception):
    def __init__(self, message, errors):
        super().__init__(message)
        self.errors = errors
