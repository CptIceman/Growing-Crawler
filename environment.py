import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

from body import Body

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
NUM_SEGMENTS = 6
MAX_SPEED = 6.0
MAX_STEPS = 500


class GrowingCrawlerEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None, manual_control=False):
        super().__init__()
        self.render_mode = render_mode
        self.manual_control = manual_control

        self.num_segments = NUM_SEGMENTS
        self.num_joints = NUM_SEGMENTS - 1

        obs_size = 2 * self.num_joints + 4
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            -MAX_SPEED, MAX_SPEED, shape=(self.num_joints,), dtype=np.float32
        )

        self.body = None
        self.screen = None
        self.clock = None
        self.steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.body = Body(num_segments=self.num_segments)
        self.steps = 0

        if self.render_mode == "human" and self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Growing Crawler")
            self.clock = pygame.time.Clock()

        return self._get_obs(), {}

    def step(self, action=None):
        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    raise SystemExit

        if self.manual_control:
            action = self._get_manual_action()

        action = np.clip(action, -MAX_SPEED, MAX_SPEED)
        self.body.apply_action(action)

        x_before = self.body.center_of_mass()[0]
        self.body.step()
        x_after = self.body.center_of_mass()[0]

        reward = float(x_after - x_before)
        self.steps += 1
        truncated = self.steps >= MAX_STEPS

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, False, truncated, {}

    def _get_manual_action(self):
        pygame.event.pump()
        keys = pygame.key.get_pressed()
        direction = 0
        if keys[pygame.K_a]:
            direction = -1
        if keys[pygame.K_d]:
            direction = 1
        action = np.zeros(self.action_space.shape, dtype=np.float32)
        action[0::2] = MAX_SPEED * direction
        action[1::2] = -MAX_SPEED * direction
        return action

    def _get_obs(self):
        angles, ang_vels = [], []
        for j in self.body.joints:
            angles.append(j.get_relative_angle())
            ang_vels.append(j.body_b.angular_velocity - j.body_a.angular_velocity)

        segs = self.body.segments
        vel = np.mean([[s.get_velocity().x, s.get_velocity().y] for s in segs], axis=0)
        cx, cy = self.body.center_of_mass()

        return np.array(
            angles + ang_vels + [cy, segs[0].get_angle(), vel[0], vel[1]],
            dtype=np.float32,
        )

    def render(self):
        if self.screen is None:
            return
        self.screen.fill((30, 30, 40))
        pygame.draw.line(
            self.screen, (100, 200, 100),
            (0, SCREEN_HEIGHT - 50), (SCREEN_WIDTH, SCREEN_HEIGHT - 50), 5
        )
        for segment in self.body.segments:
            segment.draw(self.screen, SCREEN_HEIGHT)
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None