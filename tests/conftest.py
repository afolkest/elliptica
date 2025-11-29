"""Test configuration for elliptica."""

import pytest

from elliptica.pde.register import register_all_pdes


@pytest.fixture(scope="session", autouse=True)
def register_pdes():
    """Ensure PDE registry is populated for all tests."""
    register_all_pdes()
