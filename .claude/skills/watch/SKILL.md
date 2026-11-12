---
name: watch
description: Open the pygame window and play back a trained crawler model.
disable-model-invocation: true
argument-hint: "[model path | latest]"
---

Play back a trained model in the pygame window.

Pick the model from $ARGUMENTS:
- Empty: `crawler_ppo` (the model from `train.py`).
- `latest`: the `model` field of the last line in `runs/history.jsonl`, without the `.zip`.
- Anything else: use it as the path, dropping a trailing `.zip`.

Check that `<model>.zip` exists. If it doesn't, list what's in the project root and `runs/`, then stop.

Run `.venv/bin/python watch.py <model>` in the background (`run_in_background`), because it opens a window and runs for about 80 seconds (5000 steps at 60 fps). Tell the user which model is playing and that they can close the window to stop. You can't see the window, so don't describe how the crawler moves. Ask the user what they see if it matters for the next step.
