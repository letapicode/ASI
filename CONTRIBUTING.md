# Contributing

Thank you for improving the ASI prototype. To get started:

1. Use Python 3.10 or newer.
2. Install dependencies with `pip install -r requirements.txt`.
3. Install the package in editable mode with `pip install -e .`.
4. Run tests with `pytest` before submitting a pull request.
5. Run `python scripts/security_scan.py` to check dependencies and source code for vulnerabilities.
6. Generate module docs with `python -m asi.doc_summarizer <module>` whenever you add new modules.

You can run `scripts/setup_test_env.sh` to automate steps 2 and 3.
