import argparse
import json
from asi.privacy import PrivacyBudgetManager


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show remaining privacy budget")
    parser.add_argument("log", help="Budget log file")
    parser.add_argument("--run", default="default")
    parser.add_argument("--budget", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    args = parser.parse_args()

    mgr = PrivacyBudgetManager(args.budget, args.delta, args.log)
    eps, delt = mgr.remaining(args.run)
    print(json.dumps({"epsilon": eps, "delta": delt}))
