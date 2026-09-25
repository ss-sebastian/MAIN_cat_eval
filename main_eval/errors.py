"""Project-specific exception hierarchy.

Custom types make failure modes explicit and easy to assert on in tests.
"""

from __future__ import annotations


class MainEvalError(Exception):
    """Base class for all main_eval errors."""


class DatasetError(MainEvalError):
    """Raised for malformed or invalid dataset records."""


class ChildSeparationError(MainEvalError):
    """Raised when the demonstration pool and eval set share a child."""


class SamplingError(MainEvalError):
    """Raised when demonstration sampling cannot be performed."""


class PromptError(MainEvalError):
    """Raised for missing/unreplaced prompt template placeholders."""


class AdapterError(MainEvalError):
    """Raised for model-adapter configuration / setup problems."""


class OutputExistsError(MainEvalError):
    """Raised when refusing to overwrite existing results."""
