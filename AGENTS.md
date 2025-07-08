# AGENTS Instructions

This repository contains documentation on algorithms towards self-improving AI. The main content lives in `docs/Plan.md`.

## Editing guidelines

- Use Markdown formatting.
- Avoid trailing whitespace.
- No line length limit, but keep code modular
- Reference or update `docs/Plan.md` where relevant.
- Commit messages should start with a short summary in the imperative mood, e.g. `Add section on scalability`.
- Every pull request must include a summary of what changed, how it was done, and why in `steps_summary.md`.

## Testing

Minimal unit tests live under `tests/`.
Run them with `pytest` or `python -m unittest` whenever you change code in `src/`.
Pure documentation edits can skip them.

