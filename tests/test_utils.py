"""Tests for cpas_toolbox.utils module."""
import inspect
import os
import pathlib

import pytest

from cpas_toolbox import utils


def test_str_to_object() -> None:
    """Test str_to_object function."""
    a = 1

    assert utils.str_to_object("a") is a

    assert utils.str_to_object("utils") is utils

    assert inspect.isclass(utils.str_to_object("torch.Tensor"))

    assert utils.str_to_object("____") is None


def test_resolve_path(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test resolve_path function."""
    # abs path should not be modified
    assert utils.resolve_path("/test/abs_path") == "/test/abs_path"

    # create test files
    (tmp_path / "sub1").mkdir()
    (tmp_path / "sub1" / "test.file").write_text("")
    (tmp_path / "sub2").mkdir()
    (tmp_path / "sub2" / "test.file").write_text("")

    # find directory in search path
    assert utils.resolve_path("sub1", [tmp_path]) == os.path.abspath(tmp_path / "sub1")

    # file not found in search path should return original path
    assert utils.resolve_path("test.file", [tmp_path]) == "test.file"

    # file found in second search path
    assert utils.resolve_path(
        "test.file", [tmp_path, tmp_path / "sub2"]
    ) == os.path.abspath(tmp_path / "sub2" / "test.file")

    # file found in first search path
    assert utils.resolve_path(
        "test.file", [tmp_path / "sub1", tmp_path / "sub2"]
    ) == os.path.abspath(tmp_path / "sub1" / "test.file")

    # expand home dir
    def mock_expanduser(path: str) -> str:
        return path.replace("~", "/home/name")

    monkeypatch.setattr(os.path, "expanduser", mock_expanduser)
    assert utils.resolve_path("~/test.file") == "/home/name/test.file"

    # abs path for explicit relative paths (i.e., starting with . or ..)
    def mock_abspath(path: str) -> str:
        parts = path.split(os.sep)
        if parts[0] == ".":
            return path.replace(".", "/home/name", 1)
        elif parts[0] == "..":
            return path.replace("..", "/home/name", 1)

    monkeypatch.setattr(os.path, "abspath", mock_abspath)
    assert utils.resolve_path("./test.file") == "/home/name/test.file"
    assert utils.resolve_path("../test.file") == "/home/name/test.file"
