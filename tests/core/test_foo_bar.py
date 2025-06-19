"""Integration test example."""

import pytest

from midst_toolkit.core.bar import bar as barfn
from midst_toolkit.core.foo import foo as foofn


@pytest.mark.integration_test()
def test_foofn_barfn(my_test_number: int) -> None:
    """Test foo and bar."""
    foobar = foofn("bar") + f" {my_test_number} " + barfn("foo")
    assert foobar == "foobar 42 barfoo"
