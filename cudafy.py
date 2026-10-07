#!/usr/bin/env python3
"""Unified CUDA compatibility dispatcher for cudafy-for-sail.

Three ways to run a conversion:

  python3 cudafy.py general <path>
      Convert ANY tree with the general rules only (full SDK naming map +
      compile-chain rules). Bundled actlize copies under the path are
      converted together with the tree, so no separate actlize step is needed.

  python3 cudafy.py <repo> [--version=X] <path>
      actlize, flash-attention and xformers are fully covered by the
      general rules (their per-repo scripts are gone); the command runs the
      general engine directly, keeping the historical CLI shape. deepgemm /
      flashmla additionally apply their residual rules via the per-repo
      scripts under <repo>/.

      An unknown <repo> name falls back to `general <path>`: the name is
      dropped and the path is converted with the general rules.
"""

import argparse
import importlib.util
import os
import re
import runpy
import sys
from pathlib import Path
from types import ModuleType
from typing import Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

# Repositories whose conversion still carries residual rules; everything else
# is fully covered by the general engine.
RESIDUAL_REGISTRY = {
    "deepgemm": REPO_ROOT / "deepgemm" / "cuda_compat.py",
    "flashmla": REPO_ROOT / "flashmla" / "cuda_compat.py",
}

KNOWN_COMMANDS = {
    "general", "actlize", "flash-attention", "xformers",
} | set(RESIDUAL_REGISTRY)


def _module_name(script_path: Path) -> str:
    safe_name = re.sub(r"\W+", "_", str(script_path.relative_to(REPO_ROOT)))
    return f"cudafy_for_sail_{safe_name}"


def _load_module(script_path: Path) -> ModuleType:
    if not script_path.is_file():
        raise FileNotFoundError(f"Compatibility script not found: {script_path}")

    spec = importlib.util.spec_from_file_location(_module_name(script_path), script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load compatibility script: {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_with_argv(script_path: Path, argv: Sequence[str]) -> int:
    old_argv = sys.argv[:]
    old_cwd = os.getcwd()
    sys.argv = [str(script_path), *argv]
    try:
        module = _load_module(script_path)
        if hasattr(module, "main"):
            result = module.main()
        else:
            result = runpy.run_path(str(script_path), run_name="__main__")
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        print(code, file=sys.stderr)
        return 1
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)

    return result if isinstance(result, int) else 0


def _require_target(value, name: str) -> str:
    if not value:
        raise SystemExit(f"ERROR: {name} is required.")
    return value


def _run_general(target: str, dry_run: bool, verbose: bool,
                 version: Optional[str] = None, label: str = "general") -> int:
    """Convert a path with the general engine (SDK map + compile chain)."""
    from general import convert_path

    if not os.path.exists(target):
        raise SystemExit(f"ERROR: path not found: {target}")

    print("=" * 60)
    print(f"{label}: PPU -> CUDA conversion (general rules)")
    print(f"Target: {os.path.abspath(target)}")
    if version:
        print(f"Version: {version} (informational; the general rules are "
              f"version-independent)")
    print("=" * 60)

    scanned, changed, renamed, _ = convert_path(target, dry_run=dry_run, verbose=verbose)
    print(f"  Files scanned: {scanned}")
    print(f"  Files changed: {changed}")
    print(f"  Renames:       {renamed}")
    if dry_run:
        print("\n  [DRY RUN - no files modified]")
    return 0


def run_general(args: argparse.Namespace) -> int:
    target = _require_target(args.target, "target path")
    return _run_general(target, args.dry_run, args.verbose)


def run_actlize(args: argparse.Namespace) -> int:
    target = _require_target(args.target, "actlize include directory")
    if not os.path.isdir(target):
        raise SystemExit(f"ERROR: directory not found: {target}")
    return _run_general(target, args.dry_run, args.verbose,
                        version=args.version, label="actlize")


def run_xformers(args: argparse.Namespace) -> int:
    target = _require_target(args.target, "target xformers directory")
    if (not os.path.isfile(os.path.join(target, "setup.py")) or
            not os.path.isdir(os.path.join(
                target, "xformers/csrc/attention/cuda/fmha"))):
        raise SystemExit(f"ERROR: {target} is not an xformers root directory.")
    return _run_general(target, args.dry_run, args.verbose,
                        version=args.version, label="xformers")


def run_flash_attention(args: argparse.Namespace) -> int:
    target = _require_target(args.target, "target flash-attention directory")
    if (not os.path.isfile(os.path.join(target, "setup.py")) or
            not os.path.isdir(os.path.join(target, "csrc/flash_attn"))):
        raise SystemExit(f"ERROR: {target} is not a flash-attention root directory.")
    return _run_general(target, args.dry_run, args.verbose,
                        version=args.version, label="flash-attention")


def run_residual(command: str, args: argparse.Namespace) -> int:
    script_path = RESIDUAL_REGISTRY[command]
    target = _require_target(args.target, "target repository directory")
    argv = [target]
    if args.dry_run:
        argv.append("--dry-run")
    if args.verbose:
        argv.append("--verbose")
    return _run_with_argv(script_path, argv)


def add_common_repo_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("target", help="Target repository root directory")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without modifying files")
    parser.add_argument("--verbose", action="store_true", help="Print detailed output")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run CUDA compatibility converters for supported SAIL repositories."
    )
    subparsers = parser.add_subparsers(dest="command")

    general = subparsers.add_parser(
        "general", help="Convert any path with the general rules (SDK map + compile chain)")
    add_common_repo_args(general)
    # Accepted (and ignored) so that an unknown-repo fallback carrying
    # --version=X still parses.
    general.add_argument("--version", default=None, help=argparse.SUPPRESS)
    general.set_defaults(func=run_general)

    actlize = subparsers.add_parser(
        "actlize", help="Convert ACTLIZE include files (general rules only)")
    actlize.add_argument(
        "--version", choices=["0.5.0", "0.8.0", "1.0.0"], default="1.0.0",
        help="Informational only; the general rules are version-independent")
    actlize.add_argument("target", help="Target ACTLIZE include directory")
    actlize.add_argument("--dry-run", action="store_true", help="Preview changes without modifying files")
    actlize.add_argument("--verbose", action="store_true", help="Print detailed output")
    actlize.set_defaults(func=run_actlize)

    flash_attention = subparsers.add_parser(
        "flash-attention", help="Convert a Flash-Attention repository (general rules only)")
    flash_attention.add_argument(
        "--version", choices=["2.7.2", "2.7.4", "2.8.2"], default="2.8.2",
        help="Informational only; the general rules are version-independent")
    add_common_repo_args(flash_attention)
    flash_attention.set_defaults(func=run_flash_attention)

    deepgemm = subparsers.add_parser("deepgemm", help="Convert a DeepGEMM repository")
    add_common_repo_args(deepgemm)
    deepgemm.set_defaults(func=lambda args: run_residual("deepgemm", args))

    flashmla = subparsers.add_parser("flashmla", help="Convert a FlashMLA repository")
    add_common_repo_args(flashmla)
    flashmla.set_defaults(func=lambda args: run_residual("flashmla", args))

    xformers = subparsers.add_parser(
        "xformers", help="Convert an xFormers repository (general rules only)")
    xformers.add_argument(
        "--version", choices=["0.0.27"], default="0.0.27",
        help="Informational only; the general rules are version-independent")
    add_common_repo_args(xformers)
    xformers.set_defaults(func=run_xformers)

    return parser


def main() -> int:
    argv = sys.argv[1:]
    # Unknown repo name: `cudafy.py <repo> [--version=X] <path>` falls back to
    # `cudafy.py general <path>` (the name is dropped).
    if argv and not argv[0].startswith("-") and argv[0] not in KNOWN_COMMANDS:
        print(f"NOTE: '{argv[0]}' is not a known repository; "
              f"falling back to `general` mode.")
        argv = ["general", *argv[1:]]

    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
