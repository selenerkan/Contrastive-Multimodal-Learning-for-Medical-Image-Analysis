import pytest


def test_dummy():
    assert 1 == 1


@pytest.mark.xfail()
def test_dummy_fails():
    assert 1 == 2
