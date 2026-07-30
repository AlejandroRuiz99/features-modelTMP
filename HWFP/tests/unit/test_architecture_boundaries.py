"""Architecture boundary enforcement via AST import inspection.

REQ-10: HWFP/core/ must not import from HWFP.serving.* or HWFP.training.*
REQ-11: HWFP/serving/ and HWFP/training/ must not cross-import each other.
REQ-12: HWFP/models/ and HWFP/features/ are leaf packages — they must not
        import HWFP.core, HWFP.serving, or HWFP.training.
REQ-13: HWFP/core/ must not import HWFP.models or HWFP.features (core stays
        independent of the leaf packages; adapters do the wiring).
REQ-14: nothing under HWFP/ may import legacy top-level packages
        (`src.*`, `prediction_models*`, `features_generator*`, bare
        `assembly`/`transformation`), mutate `sys.path`, or traverse via
        `Path(...).parents[3]`. Checked via AST (not text search) so that
        docstrings merely *mentioning* these patterns (e.g. explaining why
        they were removed) are never mistaken for real violations.
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


def _all_hwfp_files():
    """Every .py file under HWFP/, including conftest.py and top-level files."""
    return _HWFP_ROOT.rglob("*.py")


# REQ-14 banned import prefixes: legacy top-level packages this change absorbed
# into HWFP.models/HWFP.features. NOT included: `selection` — the design
# explicitly scopes REQ-14 to these four prefixes only (see architecture
# design D2/REQ-14); `selection` is addressed separately via dependency
# injection (HWFP.features.assembly.betting_odds.set_market_data_source),
# not an import-boundary ban.
_REQ14_BANNED_IMPORT_PREFIXES: tuple[str, ...] = (
    "src",
    "prediction_models",
    "features_generator",
    "assembly",
    "transformation",
)


def _find_sys_path_insert_calls(tree: ast.AST) -> list[ast.Call]:
    """AST-only detection of `sys.path.insert(...)` call expressions.

    Deliberately AST-based (not text search): a text search for the literal
    string "sys.path" would false-positive on docstrings that merely mention
    the pattern while explaining it was removed (see HWFP/models/paths.py).
    """
    hits: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "insert"):
            continue
        target = func.value
        if (
            isinstance(target, ast.Attribute)
            and target.attr == "path"
            and isinstance(target.value, ast.Name)
            and target.value.id == "sys"
        ):
            hits.append(node)
    return hits


def _find_parents_index_3(tree: ast.AST) -> list[ast.Subscript]:
    """AST-only detection of `<expr>.parents[3]` subscript expressions.

    Deliberately AST-based (not text search): a text search for the literal
    string "parents[3]" would false-positive on docstrings that mention the
    pattern while explaining it was removed (see HWFP/models/paths.py's
    module docstring, which documents replacing this exact pattern).
    """
    hits: list[ast.Subscript] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        value = node.value
        if not (isinstance(value, ast.Attribute) and value.attr == "parents"):
            continue
        index_node = node.slice
        # Python 3.9 wraps the subscript index in ast.Index; 3.9+ ast.Constant
        # holds the literal either way once unwrapped.
        if isinstance(index_node, ast.Index):  # pragma: no cover - py<3.9 shape
            index_node = index_node.value
        if isinstance(index_node, ast.Constant) and index_node.value == 3:
            hits.append(node)
    return hits


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


def test_models_and_features_are_leaf_packages() -> None:
    """REQ-12: HWFP/models/ and HWFP/features/ must not import HWFP.core,
    HWFP.serving, or HWFP.training — they are leaf libraries with no
    dependency on any HWFP layer.
    """
    forbidden = ("HWFP.core", "HWFP.serving", "HWFP.training")
    violations: list[tuple[str, str]] = []

    for src_dir in ("models", "features"):
        for file in _py_files(src_dir):
            for mod in _collect_imports(file):
                if any(mod == f or mod.startswith(f + ".") for f in forbidden):
                    violations.append((str(file), mod))

    assert not violations, (
        "HWFP.models/HWFP.features must stay leaf packages — violations:\n"
        + "\n".join(f"  {path}  →  {mod}" for path, mod in violations)
    )


def test_core_does_not_import_models_or_features() -> None:
    """REQ-13: HWFP/core/ must not import HWFP.models or HWFP.features —
    core stays independent of the leaf packages; adapters do the wiring.
    """
    forbidden = ("HWFP.models", "HWFP.features")
    violations: list[tuple[str, str]] = []

    for file in _py_files("core"):
        for mod in _collect_imports(file):
            if any(mod == f or mod.startswith(f + ".") for f in forbidden):
                violations.append((str(file), mod))

    assert not violations, (
        "core/ must not import the leaf packages — violations:\n"
        + "\n".join(f"  {path}  →  {mod}" for path, mod in violations)
    )


def test_no_legacy_package_imports_under_hwfp() -> None:
    """REQ-14 (imports half): zero imports of `src.*`, `prediction_models*`,
    `features_generator*`, or bare top-level `assembly`/`transformation`
    anywhere under HWFP/.
    """
    violations: list[tuple[str, str]] = []

    for file in _all_hwfp_files():
        for mod in _collect_imports(file):
            if any(
                mod == prefix or mod.startswith(prefix + ".")
                for prefix in _REQ14_BANNED_IMPORT_PREFIXES
            ):
                violations.append((str(file), mod))

    assert not violations, "Legacy top-level package imports detected:\n" + "\n".join(
        f"  {path}  →  {mod}" for path, mod in violations
    )


def test_no_sys_path_mutation_under_hwfp() -> None:
    """REQ-14 (sys.path half): zero `sys.path.insert(...)` calls anywhere
    under HWFP/, detected via AST (immune to docstring false positives).
    """
    violations: list[str] = []

    for file in _all_hwfp_files():
        tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
        if _find_sys_path_insert_calls(tree):
            violations.append(str(file))

    assert not violations, "sys.path.insert() calls detected:\n" + "\n".join(
        f"  {path}" for path in violations
    )


def test_no_parents_3_traversal_under_hwfp() -> None:
    """REQ-14 (path-traversal half): zero `<expr>.parents[3]` repo-root
    traversals anywhere under HWFP/, detected via AST (immune to docstring
    false positives — e.g. HWFP/models/paths.py's docstring documents
    *replacing* this exact pattern and must not trip the scan).
    """
    violations: list[str] = []

    for file in _all_hwfp_files():
        tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
        if _find_parents_index_3(tree):
            violations.append(str(file))

    assert not violations, "`.parents[3]` traversal detected:\n" + "\n".join(
        f"  {path}" for path in violations
    )
