"""Every module in the package must import from the DECLARED dependencies alone.

Why this test exists (2026-08-24): `handwriting_engine.line_reader` imported `cv2` and
`numpy` at module scope while neither appeared in `pyproject.toml` or `requirements.txt`.
A fresh `uv sync` therefore built an environment in which the package could not be
imported and the suite could not even be *collected* -- `pip install handwriting-engine`
produced the same ImportError for anyone else. Nothing caught it, because the only
environment it was ever run in was one where an earlier, wider install had left the
packages behind.

The check is a runtime import of every submodule rather than a lint of import
statements: an AST rule has to guess which imports are optional, while an import either
succeeds or does not. It is decisive in the CI job that installs the core dependencies
ONLY (`uv sync --no-dev` with no extras) -- there, a missing declaration is the failure.
Optional providers (torch, paddle, google-genai...) stay optional by importing their
backend inside a function or a try/except, which is exactly what makes them importable
here with the backend absent.
"""
from __future__ import annotations

import importlib
import pkgutil

import pytest

import handwriting_engine


def _submodules() -> list[str]:
    return sorted(
        m.name
        for m in pkgutil.walk_packages(handwriting_engine.__path__, "handwriting_engine.")
    )


@pytest.mark.parametrize("name", _submodules())
def test_submodule_imports_with_declared_dependencies_only(name: str) -> None:
    try:
        importlib.import_module(name)
    except ImportError as exc:  # ModuleNotFoundError is an ImportError
        pytest.fail(
            f"{name} cannot be imported: {exc}.\n"
            "Either the missing distribution belongs in pyproject.toml [project] "
            "dependencies, or the import belongs inside the function/try-except that "
            "makes it optional."
        )
