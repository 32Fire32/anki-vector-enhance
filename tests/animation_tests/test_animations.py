import os
import pytest

RUN_REAL = os.environ.get('RUN_VECTOR_ANIMATION_TESTS') == '1'


@pytest.mark.skipif(not RUN_REAL, reason='integration test: requires real Vector device and RUN_VECTOR_ANIMATION_TESTS=1')
def test_play_sample_animation():
    # This test is intentionally minimal: it exercises the runner in a controlled way.
    # It will only run if env var RUN_VECTOR_ANIMATION_TESTS=1 is set by the developer.
    from tools.run_animation_tests import load_triggers, load_observations

    triggers = load_triggers()
    assert isinstance(triggers, list)
    # Expect at least one trigger when running on-device
    assert len(triggers) > 0
