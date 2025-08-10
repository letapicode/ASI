"""Refine LLM-generated Python source via small AST transforms."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Any


class CodeRefinePipeline:
    """Apply minimal AST-level fixes to generated Python."""

    def refine(self, source: str) -> str:
        """Return refined ``source`` after applying transformations."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return source
        self._ensure_future_import(tree)
        add_any = self._annotate_functions(tree)
        self._fix_none_comparisons(tree)
        self._fix_bool_comparisons(tree)
        if add_any:
            self._ensure_import(tree, "typing", ["Any"])
        try:
            refined = ast.unparse(tree)
        except Exception:
            return source
        return refined.strip() + "\n"

    # -----------------------------------------------------
    def _annotate_functions(self, tree: ast.AST) -> bool:
        add_any = False
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                all_args = (
                    node.args.posonlyargs
                    + node.args.args
                    + node.args.kwonlyargs
                )
                if node.args.vararg:
                    all_args.append(node.args.vararg)
                if node.args.kwarg:
                    all_args.append(node.args.kwarg)
                for arg in all_args:
                    if arg.annotation is None:
                        arg.annotation = ast.Name(id="Any", ctx=ast.Load())
                        add_any = True
                if node.returns is None:
                    node.returns = ast.Name(id="Any", ctx=ast.Load())
                    add_any = True
        return add_any

    # -----------------------------------------------------
    def _fix_none_comparisons(self, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                # handle ``x == None`` and ``x != None``
                for i, (op, cmp) in enumerate(zip(node.ops, node.comparators)):
                    if isinstance(cmp, ast.Constant) and cmp.value is None:
                        if isinstance(op, ast.Eq):
                            node.ops[i] = ast.Is()
                        elif isinstance(op, ast.NotEq):
                            node.ops[i] = ast.IsNot()
                # handle ``None == x`` and ``None != x``
                if isinstance(node.left, ast.Constant) and node.left.value is None:
                    op = node.ops[0]
                    if isinstance(op, ast.Eq):
                        node.ops[0] = ast.Is()
                    elif isinstance(op, ast.NotEq):
                        node.ops[0] = ast.IsNot()
                    node.left, node.comparators[0] = node.comparators[0], node.left

    # -----------------------------------------------------
    def _fix_bool_comparisons(self, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                for i, (op, cmp) in enumerate(zip(node.ops, node.comparators)):
                    if isinstance(cmp, ast.Constant) and isinstance(cmp.value, bool):
                        if isinstance(op, ast.Eq):
                            node.ops[i] = ast.Is()
                        elif isinstance(op, ast.NotEq):
                            node.ops[i] = ast.IsNot()

    # -----------------------------------------------------
    def _ensure_import(self, tree: ast.Module, module: str, names: list[str]) -> None:
        for stmt in tree.body:
            if isinstance(stmt, ast.ImportFrom) and stmt.module == module:
                existing = {n.name for n in stmt.names}
                for name in names:
                    if name not in existing:
                        stmt.names.append(ast.alias(name=name, asname=None))
                return
        tree.body.insert(
            0,
            ast.ImportFrom(module=module, names=[ast.alias(name=n, asname=None) for n in names], level=0),
        )

    # -----------------------------------------------------
    def _ensure_future_import(self, tree: ast.Module) -> None:
        for stmt in tree.body:
            if isinstance(stmt, ast.ImportFrom) and stmt.module == "__future__":
                if any(n.name == "annotations" for n in stmt.names):
                    return
        tree.body.insert(
            0,
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations", asname=None)], level=0),
        )


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - CLI entry
    parser = argparse.ArgumentParser(description="Refine generated Python code")
    parser.add_argument("path", nargs="+", help="Python files to refine")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print refined code without modifying files",
    )
    args = parser.parse_args(argv)

    for path in args.path:
        file = Path(path)
        source = file.read_text()
        refined = CodeRefinePipeline().refine(source)
        if not args.dry_run:
            file.write_text(refined)
        print(refined)


if __name__ == "__main__":
    main()
