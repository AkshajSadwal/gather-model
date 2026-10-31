# Fix: keep the two training progress panels in place

This update changes only `scripts/run_parallel_training.py`.

## What changes

- Uses the terminal alternate screen (`1049`) for the live dashboard.
- Keeps the Model 2 and Model 1 progress panels in fixed positions.
- Refreshes the same screen rather than appending another dashboard snapshot to scrollback.
- Restores your normal terminal screen and cursor after completion or Ctrl+C.
- When output is not an interactive TTY (for example, an IDE output pane or redirected output), prints the two log paths once instead of flooding the output with repeated dashboard snapshots.

## Install

Do not stop an already-running training process just to install this display-only update. The running process has already loaded its old launcher code, so the fix applies the next time you start the launcher.

After the current run is complete, or after you intentionally stop it, open Terminal and run:

```bash
cd ~/Projects/reels-recommender-m4
unzip -o ~/Downloads/parallel-training-terminal-dashboard-fix.zip -d .
source .venv/bin/activate
python -c "import sys; print(sys.stdout.isatty())"
```

The last command should print `True` when run in the macOS Terminal app. Then start your next run with:

```bash
python scripts/run_parallel_training.py
```

Use the macOS Terminal app or another real interactive terminal. If the TTY check prints `False`, the dashboard cannot overwrite the display; the launcher will show each log path once instead. Avoid launching it in an editor's Output pane.

## Existing training configuration is unchanged

The launcher still starts both models concurrently, with Model 2 seed 42, Model 1 seed 43, batch size 4096, and maximum 20 epochs by default. This patch does not change the training scripts, model architecture, data, or checkpoints.
