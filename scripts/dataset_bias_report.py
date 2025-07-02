import argparse
import json
from pathlib import Path
from asi.dataset_bias_detector import compute_word_freq, bias_score


def main(path: str) -> None:
    freq = compute_word_freq(Path(path).glob("*.txt"))
    score = bias_score(freq)
    print(json.dumps({'bias_score': score}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dataset bias report")
    parser.add_argument('path', help='Directory of text files')
    args = parser.parse_args()
    main(args.path)
