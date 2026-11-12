---
name: train-and-eval
description: Run a short PPO training session and report whether the crawler is learning. Use after changing the reward, physics, observations, actions or hyperparameters.
disable-model-invocation: true
argument-hint: "[timesteps] [note about what changed]"
---

Quick check that a change still lets the crawler learn. This is a short diagnostic run, not a replacement for full training with `train.py`.

Arguments: $ARGUMENTS. If the first argument is a number, use it as the timestep count (default 20000, about 30s). Use the rest as the note. If there's no note, write a short one describing the uncommitted changes (`git diff --stat`).

1. Run `.venv/bin/python -m pytest -q`. If any test fails, stop and report the failure. Training a broken env wastes time.
2. Run `.venv/bin/python train_eval.py --timesteps <N> --note "<note>"`. It prints a JSON summary and appends it to `runs/history.jsonl`.
3. Report in a short table, then explain what it means:
   - **Learning curve**: `rollout_return` across `progress`. It covers only the roughly 4 episodes that finished in each rollout, so it's noisy: look at the trend, and at where it jumps (often the moment a gait is discovered). Older history entries have a cumulative `ep_rew_mean` instead, which lags and smooths. Don't compare the two directly.
   - **Trained vs random**: compare the mean of `trained_policy_returns` (deterministic, the mean action only, which is what `/watch` shows) with the mean of `random_policy_returns`. A trained policy that doesn't beat random has not learned to crawl. Returns are forward displacement in pymunk units.
   - **Noise dependence**: compare `trained_policy_noisy_returns` (with exploration noise) with `trained_policy_returns`. If the noisy score is much higher, the gait relies on the noise and the mean action alone doesn't crawl. Explain this to the user when it happens.
   - **Exploration**: `policy_std` starts at 1.0. Falling quickly toward 0 means exploration is collapsing. Staying near 1.0 is normal for short runs.
   - **Previous runs**: read earlier lines of `runs/history.jsonl` and compare with the most recent one at the same timestep count. Use its `note` and `git` fields to say what was different.
4. Give one concrete next step, and explain the RL reasoning behind it.

Things to keep in mind when interpreting the results:
- The env is deterministic and `reset` doesn't randomize, so the deterministic trained policy gets the same return in every episode. Identical `trained_policy_returns` are expected, not a bug.
- Actions are normalized to [-1, 1] and `step()` scales them by MAX_SPEED. PPO's initial Gaussian has std 1, which matches the whole range, so many early actions clip to ±1 (bang-bang). This is why noise-dependent gaits are common early in training.
- Short runs are noisy. Don't call a regression from a small difference. Suggest re-running with `--seed 1` to confirm.
- Never overwrite `crawler_ppo.zip`. Models from this skill go to `runs/<run_id>.zip`, and `/watch runs/<run_id>` plays one.
