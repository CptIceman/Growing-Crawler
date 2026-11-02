from environment import GrowingCrawlerEnv

env = GrowingCrawlerEnv(
    render_mode="human",
    manual_control=True
)

obs, info = env.reset()

for _ in range(10000):
    obs, reward, terminated, truncated, info = env.step()

env.close()


