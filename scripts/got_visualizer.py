import argparse
import json
from pathlib import Path

from asi.got_visualizer import GOTVisualizer


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Render reasoning trace")
    parser.add_argument("json", help="Path to JSON trace")
    parser.add_argument("--out", help="Output HTML file")
    args = parser.parse_args(argv)

    viz = GOTVisualizer.from_json(args.json)
    html = viz.to_html()
    if args.out:
        Path(args.out).write_text(html, encoding="utf-8")
    else:
        print(html)


if __name__ == "__main__":  # pragma: no cover - entry point
    main()
