"""Make LightGBM/XGBoost usable on macOS without a system OpenMP install.

Both link against @rpath/libomp.dylib and search Homebrew/MacPorts prefixes that
do not exist on a machine without brew. scikit-learn already ships a genuine LLVM
libomp inside its wheel, so the fix is to add that directory to each library's
rpath - pointing at the *same file* sklearn loads, rather than copying it, because
two OpenMP runtimes in one process can deadlock or crash.

Idempotent. Re-run after reinstalling lightgbm or xgboost.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SITE = Path(__file__).resolve().parents[1] / ".venv/lib/python3.13/site-packages"
TARGETS = [
    SITE / "lightgbm/lib/lib_lightgbm.dylib",
    SITE / "xgboost/lib/libxgboost.dylib",
]


def _run(*args) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True)


def main() -> int:
    omp_dir = SITE / "sklearn/.dylibs"
    if not (omp_dir / "libomp.dylib").exists():
        print(f"no libomp.dylib under {omp_dir}; install scikit-learn first")
        return 1

    for lib in TARGETS:
        if not lib.exists():
            print(f"skip   {lib.name} (not installed)")
            continue
        # Relative to the library itself, so the venv stays relocatable.
        rel = "@loader_path/" + str(Path("../..") / omp_dir.relative_to(SITE))
        existing = _run("otool", "-l", str(lib)).stdout
        if rel in existing:
            print(f"ok     {lib.name} (rpath already set)")
            continue
        add = _run("install_name_tool", "-add_rpath", rel, str(lib))
        if add.returncode != 0:
            print(f"FAIL   {lib.name}: {add.stderr.strip()}")
            return 1
        # Editing a Mach-O invalidates its signature on Apple silicon.
        _run("codesign", "--force", "--sign", "-", str(lib))
        print(f"patched {lib.name} -> {rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
