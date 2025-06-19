# Handling Pull Request Merge Conflicts

This guide expands on the section in `docs/merge_conflict.md`.
It describes how to safely merge individual task PRs when a giant
PR has already been merged into `main`.

## Steps

1. **Clone the public repository** and check out the latest `main` branch.
2. **List open pull requests** with the GitHub CLI or the API.
3. **For each PR**, fetch the branch and run a diff against `upstream/main`.
4. **If no new logic is present**, close the PR as redundant.
5. **If the PR adds improvements**, cherry-pick or reimplement them on top of `main`.
6. **Aggregate** any unique features from multiple PRs into a single update.
7. **Submit one final PR** that resolves all merge conflicts.

This approach keeps the history clean while preserving useful contributions.
