import argparse
import json
from pathlib import Path
from asi.dataset_lineage_manager import DatasetLineageManager
from asi.data_provenance_ledger import DataProvenanceLedger


def main(root: str) -> None:
    mgr = DatasetLineageManager(root)
    ledger = DataProvenanceLedger(root)
    records = [json.dumps(step.__dict__, sort_keys=True) for step in mgr.steps]
    ok = ledger.verify(records)
    print("OK" if ok else "CORRUPTED")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Verify provenance ledger")
    p.add_argument("root", help="Dataset root")
    args = p.parse_args()
    main(args.root)
