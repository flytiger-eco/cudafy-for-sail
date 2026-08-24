#!/usr/bin/env python3
"""Unified CUDA compatibility dispatcher for cudafy-for-sail."""

import argparse
import importlib.util
import os
import re
import runpy
import sys
from pathlib import Path
from types import ModuleType
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parent

SCRIPT_REGISTRY = {
    "actlize": {
        "0.5.0": REPO_ROOT / "actlize" / "cuda_compat_v0.5.0.py",
        "0.8.0": REPO_ROOT / "actlize" / "cuda_compat_v0.8.0.py",
        "1.0.0": REPO_ROOT / "actlize" / "cuda_compat_v1.0.0.py",
    },
    "deepgemm": REPO_ROOT / "deepgemm" / "cuda_compat.py",
    "flash-attention": {
        "2.7.2": REPO_ROOT / "flash-attention" / "cuda_compat_v2.7.2.py",
        "2.7.4": REPO_ROOT / "flash-attention" / "cuda_compat_v2.7.4.py",
        "2.8.2": REPO_ROOT / "flash-attention" / "cuda_compat_v2.8.2.py",
    },
    "flashmla": REPO_ROOT / "flashmla" / "cuda_compat.py",
    "xformers": {
        "0.0.27": REPO_ROOT / "xformers" / "cuda_compat_v0.0.27.py",
    },
}


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


def run_actlize(args: argparse.Namespace) -> int:
    script_path = SCRIPT_REGISTRY["actlize"][args.version]
    target = _require_target(args.target, "actlize include directory")
    return _run_with_argv(script_path, [target])


def run_deepgemm(args: argparse.Namespace) -> int:
    script_path = SCRIPT_REGISTRY["deepgemm"]
    argv = []
    if args.target:
        argv.append(args.target)
    if args.dry_run:
        argv.append("--dry-run")
    if args.verbose:
        argv.append("--verbose")
    if args.reference_dir:
        argv.extend(["--reference-dir", args.reference_dir])
    return _run_with_argv(script_path, argv)


def run_flash_attention(args: argparse.Namespace) -> int:
    script_path = SCRIPT_REGISTRY["flash-attention"][args.version]
    argv = []
    if args.target:
        argv.append(args.target)
    if args.dry_run:
        argv.append("--dry-run")
    if args.verbose:
        argv.append("--verbose")
    return _run_with_argv(script_path, argv)


def run_flashmla(args: argparse.Namespace) -> int:
    script_path = SCRIPT_REGISTRY["flashmla"]
    argv = []
    if args.target:
        argv.append(args.target)
    if args.dry_run:
        argv.append("--dry-run")
    if args.verbose:
        argv.append("--verbose")
    return _run_with_argv(script_path, argv)


def run_xformers(args: argparse.Namespace) -> int:
    script_path = SCRIPT_REGISTRY["xformers"][args.version]
    argv = []
    if args.target:
        argv.append(args.target)
    if args.dry_run:
        argv.append("--dry-run")
    if args.verbose:
        argv.append("--verbose")
    return _run_with_argv(script_path, argv)


def add_common_repo_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("target", nargs="?", help="Target repository root directory")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without modifying files")
    parser.add_argument("--verbose", action="store_true", help="Print detailed output")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run CUDA compatibility converters for supported SAIL repositories."
    )
    subparsers = parser.add_subparsers(dest="command")

    actlize = subparsers.add_parser("actlize", help="Convert ACTLIZE include files")
    actlize.add_argument(
        "--version",
        choices=sorted(SCRIPT_REGISTRY["actlize"].keys()),
        default="1.0.0",
        help="ACTLIZE compatibility script version",
    )
    actlize.add_argument("target", help="Target ACTLIZE include directory")
    actlize.set_defaults(func=run_actlize)

    deepgemm = subparsers.add_parser("deepgemm", help="Convert a DeepGEMM repository")
    add_common_repo_args(deepgemm)
    deepgemm.add_argument("--reference-dir", help="Optional original CUDA reference directory")
    deepgemm.set_defaults(func=run_deepgemm)

    flash_attention = subparsers.add_parser("flash-attention", help="Convert a Flash-Attention repository")
    flash_attention.add_argument(
        "--version",
        choices=sorted(SCRIPT_REGISTRY["flash-attention"].keys()),
        default="2.8.2",
        help="Flash-Attention compatibility script version",
    )
    add_common_repo_args(flash_attention)
    flash_attention.set_defaults(func=run_flash_attention)

    flashmla = subparsers.add_parser("flashmla", help="Convert a FlashMLA repository")
    add_common_repo_args(flashmla)
    flashmla.set_defaults(func=run_flashmla)

    xformers = subparsers.add_parser("xformers", help="Convert an xFormers repository")
    xformers.add_argument(
        "--version",
        choices=sorted(SCRIPT_REGISTRY["xformers"].keys()),
        default="0.0.27",
        help="xFormers compatibility script version",
    )
    add_common_repo_args(xformers)
    xformers.set_defaults(func=run_xformers)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
