import pytest
import os


RUN_INTEGRATION_TESTS = "THOR_INTEGRATION_TEST" in os.environ


def integration_test(fn):
    return pytest.mark.skipif(
        not RUN_INTEGRATION_TESTS,
        reason="integration_tests not enabled",
    )(fn)
