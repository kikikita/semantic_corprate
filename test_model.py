import pytest


@pytest.mark.parametrize("phrase, expected", [
    ("Hello, World!", "hello, world!"),
    ("Nobody puts Baby in the corner!", "nobody puts baby in the corner!")
])
def test_sample(phrase, expected):
    assert phrase.lower() == expected
