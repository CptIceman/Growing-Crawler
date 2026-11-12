"""Headless smoke tests for GrowingCrawlerEnv: no pygame window, no training."""

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from body import BRISTLE_GRIP, MAX_TOTAL_BEND
from environment import MAX_STEPS, GrowingCrawlerEnv


@pytest.fixture
def env():
    e = GrowingCrawlerEnv(render_mode=None)
    yield e
    e.close()


def test_passes_gymnasium_checker(env):
    # Catches API mistakes SB3 would trip over: obs dtype/shape, reset/step signatures.
    check_env(env, skip_render_check=True)


def test_reset_obs_matches_space(env):
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32
    assert isinstance(info, dict)


def test_random_episode_stays_finite_and_truncates(env):
    env.reset(seed=0)
    env.action_space.seed(0)
    for i in range(1, MAX_STEPS + 1):
        obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        # NaN/inf in obs or reward is the classic sign of a physics blow-up.
        assert np.all(np.isfinite(obs)), f"non-finite obs at step {i}"
        assert np.isfinite(reward)
        assert not terminated
        assert truncated == (i == MAX_STEPS)


def test_rewards_sum_to_forward_displacement(env):
    # Reward is per-step change in center-of-mass x, so the episode return
    # must equal total displacement. Breaks if reward shaping is changed.
    env.reset(seed=0)
    env.action_space.seed(0)
    x_start = env.body.center_of_mass()[0]
    total = sum(env.step(env.action_space.sample())[1] for _ in range(100))
    x_end = env.body.center_of_mass()[0]
    assert total == pytest.approx(x_end - x_start, abs=1e-6)


def test_worm_does_not_fall_through_ground(env):
    # Ground surface is at y=55 (segment at y=50 with radius 5), pymunk y-up.
    env.reset(seed=0)
    env.action_space.seed(0)
    for _ in range(300):
        env.step(env.action_space.sample())
    _, com_y = env.body.center_of_mass()
    assert com_y > 50


@pytest.mark.skipif(BRISTLE_GRIP < 1.0, reason="only a perfect ratchet fully blocks backsliding")
def test_grounded_segments_never_slide_backward(env):
    env.reset(seed=0)
    env.action_space.seed(0)
    for _ in range(300):
        env.step(env.action_space.sample())
        for seg, grounded in zip(env.body.segments, env.body.ground_contacts(), strict=True):
            if grounded:
                assert seg.get_velocity().x >= -1e-9


def test_total_bend_stays_within_limit(env):
    # Stops the worm curling into a ball (~2.5 rad). Constraints are solved
    # approximately, and the 1e6-force joint motors push the limit ~0.2 rad past
    # its edge, hence the 0.3 tolerance.
    env.reset(seed=0)
    env.action_space.seed(0)
    worst = 0.0
    for _ in range(300):
        env.step(env.action_space.sample())
        bend = sum(j.get_relative_angle() for j in env.body.joints)
        worst = max(worst, abs(bend))
    assert worst <= MAX_TOTAL_BEND + 0.3, f"total bend reached {worst:.2f} rad"


def test_wrong_action_length_raises(env):
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(np.zeros(env.num_joints + 1, dtype=np.float32))
