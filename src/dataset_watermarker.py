from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from PIL import Image, PngImagePlugin, UnidentifiedImageError


_TEXT_RE = re.compile(r"<WATERMARK:(.+?)>")


def add_watermark(path: str | Path, wm_id: str) -> None:
    """Embed ``wm_id`` into the file at ``path`` based on extension."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in {".txt", ".json", ".csv"}:
        text = p.read_text()
        if not text.endswith("\n"):
            text += "\n"
        text += f"<WATERMARK:{wm_id}>"
        p.write_text(text)
    elif suffix in {".png", ".jpg", ".jpeg"}:
        try:
            with Image.open(p) as img:
                info = PngImagePlugin.PngInfo()
                for k, v in img.info.items():
                    try:
                        info.add_text(k, str(v))
                    except Exception:
                        pass
                info.add_text("watermark", wm_id)
                img.save(p, pnginfo=info)
        except Exception:
            return
    elif suffix in {".wav"}:
        with open(p, "ab") as f:
            f.write(b"#WM:" + wm_id.encode() + b"\n")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def detect_watermark(path: str | Path) -> Optional[str]:
    """Return the watermark ID embedded in ``path`` if present."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in {".txt", ".json", ".csv"}:
        m = _TEXT_RE.search(p.read_text())
        return m.group(1) if m else None
    elif suffix in {".png", ".jpg", ".jpeg"}:
        try:
            with Image.open(p) as img:
                return img.info.get("watermark")
        except Exception:
            return None
    elif suffix in {".wav"}:
        data = p.read_bytes()
        idx = data.rfind(b"#WM:")
        if idx != -1:
            end = data.find(b"\n", idx)
            if end == -1:
                end = len(data)
            return data[idx + 4 : end].decode()
        return None
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


__all__ = ["add_watermark", "detect_watermark"]
