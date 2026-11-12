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
GROUND_SCREEN_Y = SCREEN_HEIGHT - 50  # ground is at world y=50; pymunk is y-up, pygame y-down
CAMERA_LEAD = SCREEN_WIDTH / 3  # keep the worm a third of the way across the screen


class GrowingCrawlerEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None, manual_control=False):
        super().__init__()
        self.render_mode = render_mode
        self.manual_control = manual_control

        self.num_segments = NUM_SEGMENTS
        self.num_joints = NUM_SEGMENTS - 1

        # joint angles + joint angular velocities + [height, head angle, vx, vy]
        # + per-segment ground contact (1/0)
        obs_size = 2 * self.num_joints + 4 + self.num_segments
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(obs_size,), dtype=np.float32
        )
        # Normalized actions: the agent outputs [-1, 1], step() scales by MAX_SPEED.
        # PPO's initial Gaussian has std 1, so this lets exploration cover the full range.
        self.action_space = spaces.Box(
            -1.0, 1.0, shape=(self.num_joints,), dtype=np.float32
        )

        self.body = None
        self.screen = None
        self.clock = None
        self.font = None
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
            self.font = pygame.font.SysFont(None, 20)

        return self._get_obs(), {}

    def step(self, action=None):
        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    raise SystemExit

        if self.manual_control:
            action = self._get_manual_action()

        action = np.clip(action, -1.0, 1.0) * MAX_SPEED
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
        action[0::2] = direction
        action[1::2] = -direction
        return action

    def _get_obs(self):
        angles, ang_vels = [], []
        for j in self.body.joints:
            angles.append(j.get_relative_angle())
            ang_vels.append(j.body_b.angular_velocity - j.body_a.angular_velocity)

        segs = self.body.segments
        vel = np.mean([[s.get_velocity().x, s.get_velocity().y] for s in segs], axis=0)
        cx, cy = self.body.center_of_mass()

        contacts = [float(c) for c in self.body.ground_contacts()]

        return np.array(
            angles + ang_vels + [cy, segs[0].get_angle(), vel[0], vel[1]] + contacts,
            dtype=np.float32,
        )

    def render(self):
        if self.screen is None:
            return
        # Camera follows the worm horizontally.
        camera_x = self.body.center_of_mass()[0] - CAMERA_LEAD

        self.screen.fill((30, 30, 40))
        pygame.draw.line(
            self.screen, (100, 200, 100),
            (0, GROUND_SCREEN_Y), (SCREEN_WIDTH, GROUND_SCREEN_Y), 5
        )
        # Distance markers so movement is visible while the camera follows:
        # a tick every 100 units, a labelled one every 500.
        first_tick = int(camera_x // 100) * 100
        for x in range(first_tick, int(camera_x + SCREEN_WIDTH) + 100, 100):
            sx = x - camera_x
            major = x % 500 == 0
            pygame.draw.line(
                self.screen, (100, 200, 100),
                (sx, GROUND_SCREEN_Y), (sx, GROUND_SCREEN_Y + (20 if major else 10)), 2
            )
            if major:
                label = self.font.render(str(x), True, (100, 200, 100))
                self.screen.blit(label, (sx + 4, GROUND_SCREEN_Y + 12))
        for segment in self.body.segments:
            segment.draw(self.screen, SCREEN_HEIGHT, camera_x)
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None