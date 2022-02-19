"""This module provides miscellaneous utility functions."""
from typing import Any

import inspect
from pydoc import locate


def str_to_object(name: str) -> Any:
    """Try to find object with a given name.

    First scope of calling function is checked for the name, then current environment
    (in which case name has to be a fully qualified name). In the second case, the
    object is imported if found.

    Args:
        name: Name of the object to resolve.
    Returns:
        The object which the provided name refers to. None if no object was found.
    """
    # check callers local variables
    caller_locals = inspect.currentframe().f_back.f_locals
    if name in caller_locals:
        return caller_locals[name]

    # check callers global variables (i.e., imported modules etc.)
    caller_globals = inspect.currentframe().f_back.f_globals
    if name in caller_globals:
        return caller_globals[name]

    # check environment
    return locate(name)
