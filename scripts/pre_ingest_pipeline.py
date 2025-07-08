import argparse
import json
from urllib.parse import urlparse
from pathlib import Path

from asi.dataset_discovery import _parse_rss
import os
from asi.dataset_bias_detector import compute_word_freq, bias_score
from asi.cross_lingual_fairness import CrossLingualFairnessEvaluator
from asi.data_ingest import CrossLingualTranslator


def analyze_dataset(root: Path) -> dict:
    freq = compute_word_freq(root.rglob("*.txt"), num_workers=os.cpu_count())
    bscore = bias_score(freq)
    langs = [p.name for p in root.iterdir() if p.is_dir()]
    stats = {l: {"1": len(list((root / l).glob("*.txt")))} for l in langs}
    if stats:
        ev = CrossLingualFairnessEvaluator(CrossLingualTranslator(langs))
        fairness = ev.evaluate(stats)
    else:
        fairness = {}
    return {"bias_score": bscore, "fairness": fairness}


def main(rss_path: str) -> None:
    text = Path(rss_path).read_text()
    dsets = _parse_rss(text, "local")
    for d in dsets:
        parsed = urlparse(d.url)
        if parsed.scheme == "file":
            root = Path(parsed.path)
            report = analyze_dataset(root)
            print(json.dumps({"dataset": d.name, **report}, indent=2))
            # Ingestion would proceed here


if __name__ == "__main__":  # pragma: no cover - CLI helper
    parser = argparse.ArgumentParser(description="Pre-ingestion analysis demo")
    parser.add_argument("rss", help="RSS feed of datasets")
    args = parser.parse_args()
    main(args.rss)
