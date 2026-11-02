import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

from body import Body

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600


class GrowingCrawlerEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode="human", manual_control=False):
        super().__init__()

        self.render_mode = render_mode
        self.manual_control = manual_control

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3,),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-2.0,
            high=2.0,
            shape=(1,),
            dtype=np.float32
        )

        self.body = None
        self.screen = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.body = Body()

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

        self.body.apply_action(action)
        self.body.step()

        obs = self._get_obs()

        if self.render_mode == "human":
            self.render()

        return obs, 0.0, False, False, {}

    def _get_manual_action(self):
        pygame.event.pump()

        action = np.zeros(self.action_space.shape, dtype=np.float32)

        keys = pygame.key.get_pressed()


        if keys[pygame.K_a]:
            action[0] = -2.0

        if keys[pygame.K_d]:
            action[0] = 2.0

        return action

    def _get_obs(self):
        segment = self.body.segments[0]

        return np.array(
            [
                segment.get_position().x,
                segment.get_position().y,
                segment.get_angle()
            ],
            dtype=np.float32
        )

    def render(self):
        self.screen.fill((30, 30, 40))

        pygame.draw.line(
            self.screen,
            (100, 200, 100),
            (0, SCREEN_HEIGHT - 50),
            (SCREEN_WIDTH, SCREEN_HEIGHT - 50),
            5
        )

        for segment in self.body.segments:
            segment.draw(self.screen, SCREEN_HEIGHT)

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.quit()