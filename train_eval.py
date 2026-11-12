"""Short PPO training run + evaluation. Used by the /train-and-eval skill.

Usage: .venv/bin/python train_eval.py [--timesteps N] [--seed S] [--note "what changed"]

Saves the model to runs/<run_id>.zip, appends a summary to runs/history.jsonl,
and prints the summary as JSON.
"""

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from environment import GrowingCrawlerEnv

RUNS = Path("runs")


class ProgressRecorder(BaseCallback):
    """Records per-rollout episode return and policy std after each rollout.

    Unlike SB3's ep_rew_mean (a running mean over the last 100 episodes), the
    return here only covers episodes that finished during this rollout, so
    sudden jumps (e.g. discovering a gait) show up when they happen.
    """

    def __init__(self):
        super().__init__()
        self.history = []
        self._returns = []

    def _on_step(self):
        for info in self.locals["infos"]:
            if "episode" in info:  # added by SB3's Monitor wrapper at episode end
                self._returns.append(info["episode"]["r"])
        return True

    def _on_rollout_end(self):
        self.history.append({
            "timesteps": self.num_timesteps,
            "episodes": len(self._returns),
            # Training return: actions include exploration noise.
            "rollout_return": round(float(np.mean(self._returns)), 2) if self._returns else None,
            # Std of the Gaussian policy; shrinking toward 0 = exploration collapsing.
            "policy_std": round(self.model.policy.log_std.exp().mean().item(), 3),
        })
        self._returns = []


def evaluate(act, episodes, seed):
    """Runs full episodes; returns per-episode returns (= forward displacement)."""
    env = GrowingCrawlerEnv(render_mode=None)
    returns = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        total, done = 0.0, False
        while not done:
            obs, r, term, trunc, _ = env.step(act(env, obs))
            total += r
            done = term or trunc
        returns.append(round(total, 2))
    env.close()
    return returns


def git_revision():
    try:
        rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "*.py"], text=True)
        return rev + ("-dirty" if dirty.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--note", default="")
    args = p.parse_args()

    RUNS.mkdir(exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + f"-s{args.seed}"

    rng = np.random.default_rng(args.seed)

    def random_action(env, obs):
        return rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)

    random_returns = evaluate(random_action, args.episodes, args.seed)

    model = PPO("MlpPolicy", GrowingCrawlerEnv(render_mode=None), seed=args.seed,
                verbose=0, device="cpu")
    recorder = ProgressRecorder()
    model.learn(total_timesteps=args.timesteps, callback=recorder)
    model_path = RUNS / run_id
    model.save(model_path)

    # Deterministic = the policy's mean action only (what watch.py shows).
    # Stochastic = mean + exploration noise, like during training. A big gap
    # means the gait depends on the noise.
    trained_returns = evaluate(
        lambda env, obs: model.predict(obs, deterministic=True)[0], args.episodes, args.seed
    )
    noisy_returns = evaluate(
        lambda env, obs: model.predict(obs, deterministic=False)[0], args.episodes, args.seed
    )

    summary = {
        "run_id": run_id,
        "note": args.note,
        "git": git_revision(),
        "timesteps": args.timesteps,
        "seed": args.seed,
        "random_policy_returns": random_returns,
        "trained_policy_returns": trained_returns,
        "trained_policy_noisy_returns": noisy_returns,
        "progress": recorder.history,
        "model": f"{model_path}.zip",
    }
    with open(RUNS / "history.jsonl", "a") as f:
        f.write(json.dumps(summary) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
