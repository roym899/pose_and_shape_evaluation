"""This module provides miscellaneous utility functions."""
import inspect
import os
from pydoc import locate
from typing import Any, List, Optional


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


def resolve_path(path: str, search_paths: Optional[List[str]] = None) -> str:
    """Resolves a path to a full absolute path based on search_paths.

    This function considers paths of 5 different cases
        /... -> absolute path, nothing todo
        ~/... -> home dir, expand user
        ./... -> relative to current directory
        ../... -> relative to current parent directory
        ... -> relative to search paths

    Current directory is not implicitly included in search paths and has to be added
    with "." if desired. Search paths are handled in first to last order and considered
    correct if file or directory exists.

    Returns original path, if file does not exist.

    Args:
        path: The path to resolve.
        search_paths:
            List of search paths to prepend relative paths which are not explicitly
            relative to current directory.
            If None, no search paths are assumed.
    """
    if search_paths is None:
        search_paths = []

    if os.path.isabs(path):
        return path

    parts = path.split(os.sep)

    if parts[0] in [".", ".."]:
        return os.path.abspath(path)
    elif parts[0] == "~":
        return os.path.expanduser(path)

    for search_path in search_paths:
        resolved_path = os.path.expanduser(os.path.join(search_path, path))
        if os.path.exists(resolved_path):
            return os.path.abspath(resolved_path)

    return path
