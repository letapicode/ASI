import argparse
from pathlib import Path
import json
from asi.secure_dataset_exchange import SecureDatasetExchange, DatasetIntegrityProof


def _load_key(hex_str: str | None) -> bytes | None:
    return bytes.fromhex(hex_str) if hex_str else None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Secure dataset exchange")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_push = sub.add_parser("push", help="Encrypt and sign a dataset")
    p_push.add_argument("directory", help="Dataset directory")
    p_push.add_argument("package", help="Output package path")
    p_push.add_argument("--key", required=True, help="AES key in hex")
    p_push.add_argument("--sign-key", help="Ed25519 private key in hex")
    p_push.add_argument("--proof-out", help="Write integrity proof to file")

    p_pull = sub.add_parser("pull", help="Decrypt and verify a dataset")
    p_pull.add_argument("package", help="Input package path")
    p_pull.add_argument("directory", help="Destination directory")
    p_pull.add_argument("--key", required=True, help="AES key in hex")
    p_pull.add_argument("--verify-key", help="Ed25519 public key in hex")
    p_pull.add_argument("--proof-in", help="Read integrity proof from file")

    args = parser.parse_args(argv)

    key = bytes.fromhex(args.key)
    if args.cmd == "push":
        sign_key = _load_key(args.sign_key)
        ex = SecureDatasetExchange(key, signing_key=sign_key)
        path, proof = ex.push(
            Path(args.directory), Path(args.package), with_proof=bool(args.proof_out)
        )
        if args.proof_out:
            Path(args.proof_out).write_text(proof.to_json())
    else:
        verify_key = _load_key(args.verify_key)
        ex = SecureDatasetExchange(key, verify_key=verify_key, require_proof=bool(args.proof_in))
        proof = None
        if args.proof_in:
            proof = DatasetIntegrityProof.from_json(Path(args.proof_in).read_text())
        ex.pull(Path(args.package), Path(args.directory), proof=proof)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
