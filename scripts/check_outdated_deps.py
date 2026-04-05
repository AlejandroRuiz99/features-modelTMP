#!/usr/bin/env python3
"""
check_outdated_deps.py — pre-commit helper

Queries PyPI and warns if pinned minimums are behind the latest release.
Exits 0 always (warn-only, never blocks commits).

Usage:
    python scripts/check_outdated_deps.py
"""
import json
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

try:
    from packaging.version import Version
except ImportError:
    print("[!] 'packaging' no instalado -- saltando check (pip install packaging)")
    sys.exit(0)

ROOT = Path(__file__).resolve().parent.parent
SKIP = {"python"}  # no son paquetes PyPI


def parse_requirements(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        m = re.match(r"^([A-Za-z0-9_\-\.]+)\s*(.*)", line)
        if m:
            name = m.group(1).lower().replace("_", "-")
            result[name] = m.group(2).strip()
    return result


def parse_pyproject(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    if not path.exists():
        return result
    in_deps = False
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s == "dependencies = [":
            in_deps = True
            continue
        if in_deps:
            if s.startswith("]"):
                break
            dep = re.sub(r'["\',]', "", s).split("#")[0].strip()
            m = re.match(r"^([A-Za-z0-9_\-\.]+)\s*(.*)", dep)
            if m:
                name = m.group(1).lower().replace("_", "-")
                result[name] = m.group(2).strip()
    return result


def extract_lower_bound(spec: str) -> Optional[str]:
    m = re.search(r">=\s*([\d\.]+)", spec)
    return m.group(1) if m else None


def fetch_latest(package: str) -> tuple[str, Optional[str]]:
    url = f"https://pypi.org/pypi/{package}/json"
    try:
        with urllib.request.urlopen(url, timeout=6) as resp:
            data = json.load(resp)
        return package, data["info"]["version"]
    except Exception:
        return package, None


def main() -> None:
    deps: dict[str, str] = {}
    deps.update(parse_requirements(ROOT / "requirements.txt"))
    deps.update(parse_pyproject(ROOT / "pyproject.toml"))  # pyproject tiene precedencia
    deps = {k: v for k, v in deps.items() if k not in SKIP}

    print(f"[deps] Consultando {len(deps)} paquetes en PyPI...")

    outdated: list[tuple[str, str, str, str]] = []
    errors: list[str] = []

    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(fetch_latest, pkg): pkg for pkg in deps}
        for future in as_completed(futures):
            pkg, latest = future.result()
            if latest is None:
                errors.append(pkg)
                continue
            spec = deps[pkg]
            lower = extract_lower_bound(spec)
            if lower:
                try:
                    if Version(latest) > Version(lower):
                        outdated.append((pkg, lower, latest, spec))
                except Exception:
                    pass

    outdated.sort(key=lambda x: x[0])

    if outdated:
        col = 36
        print(f"\n{'Paquete':<{col}} {'Minimo actual':<22} {'Ultimo en PyPI'}")
        print("-" * (col + 40))
        for pkg, lower, latest, spec in outdated:
            print(f"  {pkg:<{col - 2}} {spec:<22} ->  {latest}")
        print(f"\n  [!] {len(outdated)} paquete(s) con versiones mas nuevas (commit NO bloqueado)\n")
    else:
        print("[ok] Todas las dependencias estan al dia\n")

    if errors:
        print(f"  [i] No se pudo consultar: {', '.join(errors)}\n")

    sys.exit(0)


if __name__ == "__main__":
    main()
