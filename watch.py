from stable_baselines3 import PPO
from environment import GrowingCrawlerEnv

env = GrowingCrawlerEnv(render_mode="human")
model = PPO.load("crawler_ppo")
obs, _ = env.reset()
for _ in range(5000):
    action, _ = model.predict(obs, deterministic=True)
    obs, r, term, trunc, _ = env.step(action)
    if trunc:
        obs, _ = env.reset()