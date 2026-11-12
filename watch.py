import sys

from stable_baselines3 import PPO

from environment import GrowingCrawlerEnv

# Optional model path: `python watch.py runs/<run_id>`. Defaults to crawler_ppo.
model_path = sys.argv[1] if len(sys.argv) > 1 else "crawler_ppo"

env = GrowingCrawlerEnv(render_mode="human")
model = PPO.load(model_path)
obs, _ = env.reset()
for _ in range(5000):
    action, _ = model.predict(obs, deterministic=True)
    obs, r, term, trunc, _ = env.step(action)
    if trunc:
        obs, _ = env.reset()
