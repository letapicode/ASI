# Self Reflection

This document summarises how the Codex agent operates and how it should be used with this
repository.

Codex executes tasks in isolated cloud containers preloaded with the repository. Each task runs
independently and may read or modify files, run commands, and commit changes. Users can monitor
progress through streamed logs and terminal output.

Instructions for Codex come from this `AGENTS.md` file and any other documentation in the repo,
especially `docs/Plan.md`, which outlines algorithms needed for a self-improving AI.

When Codex completes a task it commits its work and provides citations of logs and file lines so
humans can verify the actions. Because this repository contains only documentation, no automated
tests are required by default.

Users should manually review Codex's pull requests before merging.

\nThe project vision begins with the note: "Below is a shopping list of concrete algorithmic
gaps..." as described in docs/Plan.md.
