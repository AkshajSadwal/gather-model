# Terminal progress-bar fix

The two-stage orchestrator previously merged child stderr into stdout. Since
stdout is captured through a pipe, tqdm detected a non-interactive output and
printed a fresh progress line on every refresh.

The orchestrator now leaves child stderr attached to the parent terminal,
while capturing child stdout for the per-stage log. This lets tqdm detect the
terminal and redraw one in-place progress bar. Textual stdout messages remain
captured in the stage log and streamed by the orchestrator.

No model, data, optimizer, seed, batch-size, or evaluation behaviour changes.
