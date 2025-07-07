#!/usr/bin/env python
"""Compute differences between two reasoning graphs."""

from __future__ import annotations

import argparse
import json

from asi.reasoning_history import _diff_graph_data


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diff two GraphOfThought JSON files")
    parser.add_argument("old_graph", help="Path to the older graph JSON")
    parser.add_argument("new_graph", help="Path to the newer graph JSON")
    args = parser.parse_args()

    old = load_json(args.old_graph)
    new = load_json(args.new_graph)
    diff = _diff_graph_data(old, new)

    print(f"Added nodes: {len(diff['added_nodes'])}")
    if diff["added_nodes"]:
        print(" - " + "\n - ".join(n.get("text", "") for n in diff["added_nodes"]))
    print(f"Changed nodes: {len(diff['changed_nodes'])}")
    print(f"Added edges: {len(diff['added_edges'])}")
    print(f"Changed edges: {len(diff['changed_edges'])}")


if __name__ == "__main__":  # pragma: no cover - CLI helper
    main()
