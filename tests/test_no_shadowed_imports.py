"""
Guard against a function-local `import x` that shadows a module-level name used
earlier in the same function. Python then treats `x` as local for the whole
function, so the earlier use raises UnboundLocalError at runtime
(e.g. "cannot access local variable 'asyncio'").
"""
import ast
import pathlib

APP = pathlib.Path(__file__).resolve().parent.parent / "app"


def test_no_local_import_used_before_definition():
    bugs = []
    for path in APP.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            local_imports = {}
            for node in ast.walk(fn):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    for alias in node.names:
                        name = alias.asname or alias.name.split(".")[0]
                        local_imports.setdefault(name, node.lineno)
            for name, line in local_imports.items():
                early = [n.lineno for n in ast.walk(fn)
                         if isinstance(n, ast.Name) and n.id == name and n.lineno < line]
                if early:
                    bugs.append(f"{path.name}:{fn.name} uses {name!r} on line {early[0]} before local import on line {line}")
    assert not bugs, "\n".join(bugs)
