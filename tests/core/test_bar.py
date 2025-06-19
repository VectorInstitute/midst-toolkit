"""Test for bar module."""

from midst_toolkit.core.bar import bar as barfn


def test_barfn(my_test_number: int) -> None:
    """Test bar function."""
    assert barfn(f"foo-{my_test_number}") == "barfoo-42"
