# Microcontroller Inference

This note describes how to export the tiny models for deployment on
microcontrollers using either TFLite‑Micro or microTVM.

The models are first exported to ONNX just like in
[`docs/Implementation.md`](Implementation.md) which mentions
`scripts/export_onnx.py`. WASM export for browsers is covered in
[`scripts/export_wasm.py`](../scripts/export_wasm.py) and the plan notes in
[`docs/Plan.md`](Plan.md).

## Exporting

Run `scripts/export_micro.py` to produce both TFLite and microTVM binaries:

```bash
python scripts/export_micro.py --out-dir micro_models
```

This will create `*.tflite` and `*_micro.tar` files next to the intermediate
ONNX graphs.

## Loading

At runtime the binaries can be loaded by the chosen micro inference engine.
`export_to_tflite_micro()` and `export_to_microtvm()` fall back to simple copies
when the necessary toolchains are missing so tests remain lightweight. When the
tools are available, the resulting files can be flashed onto a board using the
corresponding SDKs.

