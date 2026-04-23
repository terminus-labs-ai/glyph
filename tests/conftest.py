import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--reranker-url",
        action="store",
        default=None,
        help="URL for live reranker integration tests (e.g. http://localhost:11434). Skips if not set.",
    )
