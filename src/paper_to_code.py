import re
from pathlib import Path


def _strip_latex(line: str) -> str:
    """Remove basic LaTeX commands and math wrappers."""
    line = re.sub(r"\\(?:begin|end)\{[^}]*\}", "", line)
    line = re.sub(r"\\(?:caption|label)\{[^}]*\}", "", line)
    line = line.replace("$$", "")
    line = line.replace("$", "")
    for wrapper in ["\\(", "\\)", "\\[", "\\]"]:
        line = line.replace(wrapper, "")
    return line.strip()


def transpile(latex_code: str) -> str:
    """Transpile simple LaTeX pseudo-code to Python code."""
    lines = [_strip_latex(l) for l in latex_code.splitlines()]
    py_lines = []
    indent = 0
    for line in lines:
        if not line:
            continue
        if line.strip() == "}":
            indent = max(indent - 1, 0)
            continue
        if re.match(r"\\End(For|If|While|Function|Procedure)", line):
            indent = max(indent - 1, 0)
            continue
        elif line.startswith("\\ElseIf"):
            indent = max(indent - 1, 0)
            cond = re.search(r"\\ElseIf\{([^}]*)\}", line)
            py_lines.append("    " * indent + f"elif {cond.group(1)}:")
            indent += 1
            continue
        elif line.startswith("\\Else"):
            indent = max(indent - 1, 0)
            py_lines.append("    " * indent + "else:")
            indent += 1
            continue
        m = re.search(r"\\(Function|Procedure)\{([^}]*)\}(?:\{([^}]*)\})?", line)
        if m:
            name = m.group(2)
            args = m.group(3) or ""
            if args:
                py_lines.append("    " * indent + f"def {name}({args}):")
            else:
                py_lines.append("    " * indent + f"def {name}:")
            indent += 1
            continue
        if line.startswith("\\Repeat"):
            py_lines.append("    " * indent + "while True:")
            indent += 1
            continue
        m = re.search(r"\\For\{([^}]*)\}", line)
        if m:
            py_lines.append("    " * indent + f"for {m.group(1)}:")
            indent += 1
            continue
        m = re.search(r"\\If\{([^}]*)\}", line)
        if m:
            py_lines.append("    " * indent + f"if {m.group(1)}:")
            indent += 1
            continue
        m = re.search(r"\\While\{([^}]*)\}", line)
        if m:
            py_lines.append("    " * indent + f"while {m.group(1)}:")
            indent += 1
            continue
        m = re.search(r"\\Until\{([^}]*)\}", line)
        if m:
            py_lines.append("    " * indent + f"if {m.group(1)}:")
            py_lines.append("    " * (indent + 1) + "break")
            indent = max(indent - 1, 0)
            continue
        m = re.search(r"\\Return\{?([^}]*)\}?", line)
        if m:
            py_lines.append("    " * indent + f"return {m.group(1)}")
            continue
        line = re.sub(r"\\State(?:\{([^}]*)\})?", lambda m: m.group(1) or "", line)
        line = line.lstrip()
        line = re.sub(r"\\Comment\{([^}]*)\}", lambda mo: "# " + mo.group(1), line)
        if line:
            py_lines.append("    " * indent + line)
    return "\n".join(py_lines)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Transpile LaTeX pseudo-code to Python")
    parser.add_argument("input", help="Path to LaTeX file")
    parser.add_argument("-o", "--output", default="out.py", help="Output Python file")
    args = parser.parse_args()

    text = Path(args.input).read_text()
    code = transpile(text)
    Path(args.output).write_text(code)


if __name__ == "__main__":
    main()
