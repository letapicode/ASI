import argparse
from pathlib import Path
from asi.p2p_dataset_exchange import P2PDatasetExchange, FileDHT


def _load_key(hex_str: str | None) -> bytes | None:
    return bytes.fromhex(hex_str) if hex_str else None


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="P2P dataset exchange")
    sub = p.add_subparsers(dest="cmd", required=True)

    push = sub.add_parser("push", help="Encrypt and share a dataset")
    push.add_argument("root", help="Exchange root directory")
    push.add_argument("dataset_id", help="Dataset identifier")
    push.add_argument("directory", help="Dataset directory")
    push.add_argument("--key", required=True, help="AES key in hex")
    push.add_argument("--sign-key", help="Ed25519 private key in hex")
    push.add_argument("--chunk-size", type=int, default=1_048_576)

    seed = sub.add_parser("seed", help="Run a simple DHT server")
    seed.add_argument("root", help="Exchange root directory")
    seed.add_argument("--port", type=int, default=8765)

    pull = sub.add_parser("pull", help="Retrieve and decrypt a dataset")
    pull.add_argument("root", help="Exchange root directory")
    pull.add_argument("dataset_id", help="Dataset identifier")
    pull.add_argument("directory", help="Destination directory")
    pull.add_argument("--key", required=True, help="AES key in hex")
    pull.add_argument("--verify-key", help="Ed25519 public key in hex")

    args = p.parse_args(argv)
    dht = FileDHT(Path(args.root) / "dht")

    if args.cmd == "push":
        ex = P2PDatasetExchange(
            args.root,
            dht,
            bytes.fromhex(args.key),
            signing_key=_load_key(args.sign_key),
        )
        ex.push(Path(args.directory), args.dataset_id, chunk_size=args.chunk_size)
    elif args.cmd == "pull":
        ex = P2PDatasetExchange(
            args.root,
            dht,
            bytes.fromhex(args.key),
            verify_key=_load_key(args.verify_key),
        )
        ex.pull(args.dataset_id, Path(args.directory))
    else:  # seed
        from aiohttp import web

        async def handle_get(request: web.Request) -> web.Response:
            key = request.match_info["key"]
            data = dht.get(key)
            if data is None:
                raise web.HTTPNotFound()
            return web.Response(body=data)

        async def handle_put(request: web.Request) -> web.Response:
            key = request.match_info["key"]
            data = await request.read()
            dht.put(key, data)
            return web.Response(text="OK")

        app = web.Application()
        app.add_routes([web.get("/{key}", handle_get), web.put("/{key}", handle_put)])
        web.run_app(app, port=args.port)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
