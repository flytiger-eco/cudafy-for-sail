#!/usr/bin/env python3
"""FlashMLA conversion -- general engine + FlashMLA-specific residuals.

The general engine (full SDK naming map + compile-chain rules) covers
everything the old FlashMLA script did except three structural rules that
cannot be expressed as token renames; they are applied AFTER the general
conversion and are the complete residual rule set:

  1. csrc/api/common.h forward-declares hggcStream_t because the PPU SDK
     headers are not on the host compiler's include path; the compat build
     gets cudaStream_t from cuda_runtime.h via torch, so the (converted)
     typedef must be removed instead of renamed -- renaming it would collide.
  2. setup.py must link the CUDA driver library (libraries=['cuda']) for the
     driver-AD launch path.
  3. setup.py enables the ptxas resource report and the register-usage-level
     build tuning.

Everything else (hggc/hgtx/HGAD token renames, arch flags -> single
compute_80/sm_80 gencode, hgcc -> nvcc identifiers) is handled by the
general engine.

Usage (unchanged):
    python cuda_compat.py [TARGET_DIR] [--dry-run] [--verbose]
"""

import argparse
import os
import shutil
import sys
import tempfile
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from general import convert_path  # noqa: E402

# --- residual 1: the forward-declared stream typedef must disappear --------
# Anchored on the post-general text (the general engine already renamed
# hggcStream_t -> cudaStream_t and HGstream_st -> CUstream_st).
FORWARD_DECLARE = (
    "// Forward-declare cudaStream_t so function signatures match between .cu and .cpp\n"
    "typedef struct CUstream_st* cudaStream_t;\n"
)

# --- residual 2/3: setup.py build additions (idempotent) -------------------
DRIVER_LIB_ANCHOR = (
    "        sources=get_sources(),\n"
    "        extra_compile_args={\n"
)
DRIVER_LIB_INSERT = (
    "        sources=get_sources(),\n"
    "        libraries=['cuda'],\n"
    "        extra_compile_args={\n"
)
PTXAS_ANCHOR = (
    '                    "--use_fast_math",\n'
    '                    "-mllvm",\n'
)
PTXAS_INSERT = (
    '                    "--use_fast_math",\n'
    '                    "--ptxas-options=-v,--register-usage-level=10",\n'
    '                    "-mllvm",\n'
)


def _read_text(filepath: str):
    try:
        with open(filepath, "r", encoding="utf-8", errors="surrogateescape",
                  newline="") as f:
            return f.read()
    except OSError as exc:
        print(f"  [residual] WARNING: cannot read {filepath}: {exc}")
        return None


def _write_text(filepath: str, content: str) -> bool:
    try:
        with open(filepath, "w", encoding="utf-8", errors="surrogateescape",
                  newline="") as f:
            f.write(content)
    except OSError as exc:
        print(f"  [residual] WARNING: cannot write {filepath}: {exc}")
        return False
    return True


def apply_residuals(target_dir: str, dry_run: bool, verbose: bool) -> int:
    """Apply the FlashMLA-specific residual rules; returns files changed."""
    changed = 0

    common_h = os.path.join(target_dir, "csrc", "api", "common.h")
    if os.path.isfile(common_h):
        content = _read_text(common_h)
        if content is not None and FORWARD_DECLARE in content:
            if verbose:
                print("  [residual] drop forward-declared cudaStream_t "
                      "(csrc/api/common.h)")
            converted = content.replace(FORWARD_DECLARE, "")
            if dry_run or _write_text(common_h, converted):
                changed += 1

    setup_py = os.path.join(target_dir, "setup.py")
    if os.path.isfile(setup_py):
        content = _read_text(setup_py)
        if content is None:
            return changed
        original = content
        if "libraries=['cuda']" not in content and DRIVER_LIB_ANCHOR in content:
            if verbose:
                print("  [residual] link the CUDA driver library (setup.py)")
            content = content.replace(DRIVER_LIB_ANCHOR, DRIVER_LIB_INSERT)
        if "--ptxas-options" not in content and PTXAS_ANCHOR in content:
            if verbose:
                print("  [residual] enable ptxas resource report (setup.py)")
            content = content.replace(PTXAS_ANCHOR, PTXAS_INSERT)
        if content != original and (dry_run or _write_text(setup_py, content)):
            changed += 1

    return changed


def _preview_tree(target_dir: str):
    preview = tempfile.TemporaryDirectory(prefix="cudafy-flashmla-")
    work_dir = os.path.join(preview.name, "repo")
    shutil.copytree(
        target_dir, work_dir,
        ignore=shutil.ignore_patterns(".git", "__pycache__", "build", "dist",
                                      ".eggs", "*.egg-info"),
    )
    return preview, work_dir


def main():
    parser = argparse.ArgumentParser(
        description='FlashMLA PPU-original -> CUDA-Compatible Transform '
                    '(general engine + residuals)')
    parser.add_argument('target_dir', nargs='?', default='.',
                        help='Target FlashMLA root directory')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview without modifying files')
    parser.add_argument('--verbose', action='store_true',
                        help='Print each modified file')
    args = parser.parse_args()

    target_dir = os.path.abspath(args.target_dir)
    if (not os.path.isfile(os.path.join(target_dir, 'setup.py')) or
            not os.path.isdir(os.path.join(target_dir, 'csrc'))):
        print(f"ERROR: {target_dir} is not a FlashMLA root directory.")
        return 1

    print("=" * 60)
    print("FlashMLA PPU-original -> CUDA-Compatible Transform")
    print(f"Target: {target_dir}")
    print("=" * 60)

    t0 = time.time()
    preview = None
    work_dir = target_dir
    if args.dry_run:
        preview, work_dir = _preview_tree(target_dir)
        print("  Preview runs against an isolated temporary copy.")

    try:
        print("\n[1/2] General engine (full SDK map + compile chain)...")
        scanned, changed, renamed, _ = convert_path(
            work_dir, dry_run=False, verbose=args.verbose)
        print(f"  Files scanned: {scanned}")
        print(f"  Files changed: {changed}")
        print(f"  Renames:       {renamed}")

        print("\n[2/2] FlashMLA residual rules...")
        n_residual = apply_residuals(work_dir, False, args.verbose)
        print(f"  Residual changes: {n_residual}")

        print(f"\nDone in {time.time() - t0:.1f}s")
        print("=" * 60)

        if args.dry_run:
            print("\n  [DRY RUN - no source files modified]")
        else:
            print("\n  Next steps:")
            print("  1. Build: python setup.py bdist_wheel")
            print("  2. Install: pip install dist/flash_mla-*.whl --force-reinstall --no-deps")
            print("  3. Test: python -m pytest tests/test_flash_mla.py -v")
        return 0
    finally:
        if preview is not None:
            preview.cleanup()


if __name__ == '__main__':
    sys.exit(main())
