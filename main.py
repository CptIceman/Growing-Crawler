from environment import GrowingCrawlerEnv

env = GrowingCrawlerEnv(
    render_mode="human",
    manual_control=True
)

obs, info = env.reset()
total = 0
for i in range(10000):
    obs, reward, terminated, truncated, info = env.step()
    total += reward
    if i % 60 == 0:
        print(f"Step: {i}, Total Reward: {total:.2f}")
env.close()


