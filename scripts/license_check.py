import argparse
import json
from asi.license_inspector import LicenseInspector


def main(path: str) -> None:
    insp = LicenseInspector()
    results = insp.inspect_dir(path)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check dataset licenses")
    parser.add_argument("path", help="Directory with metadata JSON files")
    args = parser.parse_args()
    main(args.path)
