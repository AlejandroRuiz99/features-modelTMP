"""Architecture boundary enforcement via AST import inspection.

REQ-10: HWFP/core/ must not import from HWFP.serving.* or HWFP.training.*
REQ-11: HWFP/serving/ and HWFP/training/ must not cross-import each other.
"""

from __future__ import annotations

import ast
from pathlib import Path

# HWFP/ directory — test lives at HWFP/tests/unit/test_architecture_boundaries.py
_HWFP_ROOT: Path = Path(__file__).resolve().parents[2]

# REQ-11 sanctioned exception: {path-relative-to-_HWFP_ROOT → allowed import prefixes}
# WHY: training composition root wires FakeModelRegistry (serving/fakes) as shared fake
#      infrastructure; design explicitly approves this single cross-layer coupling.
#      Batch 5 extends this: container_production_training() wires the REAL
#      FilesystemModelRegistry (design D3 — "registry" in the production wiring list)
#      so trained candidates register into the same checkpoints tree the serving
#      layer reads production models from. Narrowly scoped to that one module, not
#      all of HWFP.serving.adapters.
_CROSS_LAYER_ALLOWLIST: dict[str, set[str]] = {
    "training/composition/container.py": {
        "HWFP.serving.fakes",
        "HWFP.serving.adapters.filesystem_model_registry",
    },
}


def _pkg_parts(file: Path) -> list[str]:
    """Package parts for relative-import resolution, e.g. ['HWFP', 'core', 'domain']."""
    rel = file.relative_to(_HWFP_ROOT.parent)  # relative to repo root
    return list(rel.with_suffix("").parts)[:-1]


def _resolve_relative(module: str | None, level: int, pkg: list[str]) -> str:
    """Convert a relative import to its absolute dotted module name."""
    steps_up = level - 1
    base = pkg[: max(0, len(pkg) - steps_up)]
    if module:
        return ".".join(base + [module])
    return ".".join(base)


def _collect_imports(file: Path) -> list[str]:
    """Return all absolute module strings referenced by every import in *file*."""
    tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
    pkg = _pkg_parts(file)
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                # Relative import — resolve to absolute before checking
                modules.append(_resolve_relative(node.module, node.level, pkg))
            elif node.module:
                modules.append(node.module)
    return modules


def _py_files(subdir: str):
    return (_HWFP_ROOT / subdir).rglob("*.py")


def test_core_does_not_import_outer_layers() -> None:
    """REQ-10: nothing inside HWFP/core/ may import HWFP.serving.* or HWFP.training.*"""
    forbidden = ("HWFP.serving", "HWFP.training")
    violations: list[tuple[str, str]] = []

    for file in _py_files("core"):
        for mod in _collect_imports(file):
            if any(mod == f or mod.startswith(f + ".") for f in forbidden):
                violations.append((str(file), mod))

    assert not violations, (
        "core/ must not import outer layers — violations:\n"
        + "\n".join(f"  {path}  →  {mod}" for path, mod in violations)
    )


def test_serving_and_training_isolated() -> None:
    """REQ-11: HWFP/serving/ must not import HWFP.training.*, and vice versa.

    One sanctioned exception: training/composition/container.py may import
    HWFP.serving.fakes (see _CROSS_LAYER_ALLOWLIST and its WHY comment).
    """
    checks: list[tuple[str, str]] = [
        ("serving", "HWFP.training"),
        ("training", "HWFP.serving"),
    ]
    violations: list[tuple[str, str, str]] = []

    for src_dir, forbidden_prefix in checks:
        for file in _py_files(src_dir):
            rel_key = file.relative_to(_HWFP_ROOT).as_posix()
            allowed: set[str] = _CROSS_LAYER_ALLOWLIST.get(rel_key, set())
            for mod in _collect_imports(file):
                if mod == forbidden_prefix or mod.startswith(forbidden_prefix + "."):
                    if not any(mod == a or mod.startswith(a + ".") for a in allowed):
                        violations.append((src_dir, str(file), mod))

    assert not violations, "Sibling-layer cross-imports detected:\n" + "\n".join(
        f"  [{src}] {path}  →  {mod}" for src, path, mod in violations
    )
