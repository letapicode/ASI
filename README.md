# ASI Prototype

This repository experiments with algorithms needed for self-improving AI. The big picture lives in
[docs/Plan.md](docs/Plan.md), which outlines scaling-efficiency, long-context, and alignment tasks.

## Prototype Scripts

- `scripts/benchmark_moe.py` measures parameter counts and FLOPs. It can use a Mixture-of-Experts router.
- `scripts/moe_vs_dense.py` benchmarks dense versus Mixture-of-Experts feed-forward layers.
- `python -m src.paper_to_code` transpiles LaTeX pseudo-code to Python.
- `python -m src.autobench` runs each test file in isolation and reports a summary.
- `meta-rl-refactor` parses action/reward logs and suggests the next refactoring step.
- `scripts/dataset_summary.py` prints lineage and license info. Use `--content` to cluster dataset samples and store summaries under `docs/datasets/`.
- `scripts/lineage_viewer.py <root>` serves an interactive graph of the dataset lineage.
- `scripts/ar_robot_demo.py` streams predicted and actual robot trajectories to a WebSocket server for lightweight AR visualization. Pass `--show-graph` to also broadcast `GraphOfThought` nodes.

Example:

```bash
meta-rl-refactor sample_log.csv
```

## Setup

1. Use Python 3.10 or newer with PyTorch installed.
2. Install dependencies with `pip install -r requirements.txt` or run
   `scripts/setup_test_env.sh` to automate the process.
3. Optional: `pip install flash-attn` to enable the FlashAttention-3 wrapper in `src/flash_attention3.py`.
4. Optional: `pip install faiss-cpu` to enable disk-backed vector storage in `src/vector_store.py`.
5. Run `pip install -e .` to enable imports from the `asi` package.
6. The project runs without these optional packages, but FlashAttention-3 and persistent storage will be disabled.
7. Launch the AR demo with `python scripts/ar_robot_demo.py` (add `--show-graph` to stream the reasoning graph) and connect your AR client to `ws://localhost:8765/ws`.

Run the scripts directly with `python` to see parameter and FLOP estimates.

### Browser inference

Use `scripts/export_wasm.py` to create WebAssembly bundles from the example
models:

```bash
python scripts/export_wasm.py
```

Serve the generated files along with `onnxruntime-web` and load them in the
browser:

```html
<script src="node_modules/onnxruntime-web/dist/ort.wasm.min.js"></script>
<script type="module">
  import * as ort from 'onnxruntime-web';
  const session = await ort.InferenceSession.create('wasm_models/world_model.onnx');
  const output = await session.run({/* inputs */});
</script>
```

## Telemetry

`MemoryServer` can record resource usage via `TelemetryLogger`. Metrics start
and stop automatically with the server:

```python
from asi.hierarchical_memory import HierarchicalMemory
from asi.memory_service import serve
from asi.telemetry import TelemetryLogger

mem = HierarchicalMemory(dim=4, compressed_dim=2, capacity=10)
logger = TelemetryLogger(port=8000)
server = serve(mem, "localhost:50070", telemetry=logger)
```

To search across languages, wrap the memory with `CrossLingualMemory` and
provide a `CrossLingualTranslator`:

```python
from asi.cross_lingual_memory import CrossLingualMemory
from asi.data_ingest import CrossLingualTranslator

translator = CrossLingualTranslator(["es"])
mem = CrossLingualMemory(dim=4, compressed_dim=2, capacity=10, translator=translator)
mem.add("hello")
mem.search("[es] hello")
```

Visit `http://localhost:8000` to view Prometheus metrics.

`RiskDashboard` combines these metrics with ethical risk scores from
`RiskScoreboard` and serves them via the same HTTP interface. Launch it with
`scripts/memory_dashboard.py`. The script also starts an
`InterpretabilityDashboard` on the next port to display attention heatmaps and
memory statistics in a simple web UI.

### Collaboration Portal

`CollaborationPortal` exposes active tasks, telemetry metrics and reasoning logs
through a lightweight web server.

```python
from asi.collaboration_portal import CollaborationPortal
from asi.telemetry import TelemetryLogger
from asi.reasoning_history import ReasoningHistoryLogger

tel = TelemetryLogger()
tel.start()
hist = ReasoningHistoryLogger()
portal = CollaborationPortal(["refactor repo"], tel, hist)
portal.start(port=8090)
```

Visit `http://localhost:8090` to inspect progress.

### Graph UI Editing

`GraphUI` visualizes reasoning graphs. With `NLGraphEditor` you can type plain
English commands into the text box at `/graph` to modify the graph. Examples:

```text
add node analysis
add edge from start to analysis
merge nodes analysis and finish
```

Each command triggers a recomputation of the concise summary stored in
`ReasoningHistoryLogger`. See [docs/Plan.md](docs/Plan.md) for the roadmap.

## Testing

1. Install requirements: `pip install -r requirements.txt` (or run
   `scripts/setup_test_env.sh`).
2. Install the package in editable mode: `pip install -e .` (already done by
   the setup script).
3. Run tests with `pytest`.

## Style

This project imposes no strict line-length limit. The AGENTS guidelines instead emphasize writing modular
code rather than enforcing a specific maximum line length.

## CI

Automated tests run on GitHub Actions. The workflow installs the project in editable mode and executes `pytest`.
See [.github/workflows/test.yml](.github/workflows/test.yml) for details.

Continuous evaluation runs `scripts/continuous_eval.py` after each pull request.
The script invokes `eval_harness` and `autobench` to track benchmark progress.
See [.github/workflows/continuous_eval.yml](.github/workflows/continuous_eval.yml)
for the GitHub Actions setup or schedule it locally with `cron`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on running the setup script
and tests before opening a pull request.
