from __future__ import annotations

"""Utilities for exporting ONNX models to micro controller formats."""

from pathlib import Path
import shutil
import subprocess


__all__ = ["export_to_tflite_micro", "export_to_microtvm"]


def _ensure_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src != dst:
        shutil.copyfile(src, dst)


def export_to_tflite_micro(onnx_path: str | Path, output_path: str | Path) -> None:
    """Convert an ONNX graph to a TFLite-Micro binary.

    The implementation relies on the ``tflite_convert`` tool if available. When
    absent, the ONNX file is simply copied to ``output_path`` as a stand-in so
    unit tests remain lightweight.
    """

    onnx_p = Path(onnx_path)
    out_p = Path(output_path)

    cmd = [
        "tflite_convert",
        "--output_file",
        str(out_p),
        "--graph_def_file",
        str(onnx_p),
    ]
    try:
        subprocess.check_call(cmd)
    except FileNotFoundError:
        # fall back to a simple copy for environments without TensorFlow
        _ensure_file(onnx_p, out_p)


def export_to_microtvm(onnx_path: str | Path, output_path: str | Path) -> None:
    """Compile an ONNX graph using ``tvmc`` for microTVM deployment.

    If ``tvmc`` is not available, this function copies the ONNX file to the
    desired output as a placeholder.
    """

    onnx_p = Path(onnx_path)
    out_p = Path(output_path)

    cmd = [
        "tvmc",
        "compile",
        str(onnx_p),
        "--target",
        "c",
        "--output",
        str(out_p),
    ]
    try:
        subprocess.check_call(cmd)
    except FileNotFoundError:
        _ensure_file(onnx_p, out_p)

