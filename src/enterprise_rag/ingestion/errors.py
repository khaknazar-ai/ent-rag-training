"""Fatal resource containment failures must retire the current worker."""


class ParserContainmentError(RuntimeError):
    """Parser termination or receiver cleanup could not be confirmed."""
