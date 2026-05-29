"""Top-level package for dottxt."""

from importlib.metadata import PackageNotFoundError, version

from dottxt.client import AsyncDotTxt, DotTxt, InvalidOutputError
from dottxt.streaming import PatchEvent, PatchStreamError, apply_add

try:  # pragma: no cover
    __version__ = version("dottxt")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = [
    "DotTxt",
    "AsyncDotTxt",
    "InvalidOutputError",
    "PatchEvent",
    "PatchStreamError",
    "apply_add",
]
