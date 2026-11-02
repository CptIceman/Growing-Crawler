from stable_baselines3 import PPO
from environment import GrowingCrawlerEnv

env = GrowingCrawlerEnv(render_mode=None)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=200_000)
model.save("crawler_ppo")