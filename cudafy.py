#!/usr/bin/env python3
"""Unified CUDA compatibility dispatcher for cudafy-for-sail.

Three ways to run a conversion:

  python3 cudafy.py general <path>
      Convert ANY tree with the general rules only (full SDK naming map +
      compile-chain rules). Bundled actlize copies under the path are
      converted together with the tree, so no separate actlize step is needed.

  python3 cudafy.py <repo> [--version=X] <path>
      actlize, flash-attention, xformers, deepgemm and flashmla are fully
      covered by the general rules (their per-repo scripts are gone); the
      command runs the general engine directly, keeping the historical CLI shape.

      An unknown <repo> name falls back to `general <path>`: the name is
      dropped and the path is converted with the general rules.

Additional library compatibility commands:

  python3 cudafy.py actlize --compat --version=0.8.0|1.0.0 <include>
      Opt into versioned library rules in a shared read-only plan. Without
      --compat, all versions retain the original SDK-only path. Version 0.5.0
      remains SDK-only even when compatibility is explicitly requested.

  python3 cudafy.py flex-flash-attention --version=2.8.2 <root>
      Convert outer SDK sources and generate ACTLIZE 1.0 compatibility artifacts
      for repository-local include/util trees. Never modify external ACTLIZE_DIR
      libraries automatically. Dry runs do not write conversion artifacts.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

KNOWN_COMMANDS = {
    "general", "actlize", "flash-attention", "xformers", "deepgemm", "flashmla",
}


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


def _run_compatible(target, version, dry_run, verbose, flex=False):
    from general import plan_path
    from compat import extend_actlize_plan

    target = os.path.abspath(target)
    try:
        if flex:
            flattened = (os.path.isfile(os.path.join(target, "build_lib.py"))
                         and os.path.isdir(os.path.join(target, "instantiations")))
            nested = (os.path.isfile(os.path.join(target, "setup.py"))
                      and os.path.isdir(os.path.join(target, "csrc", "flash_attn")))
            if not (flattened or nested):
                raise ValueError(f"Not a supported flex root directory: {target}")
            actlize = os.path.join(target, "csrc", "actlize")
            include = os.path.join(actlize, "include")
            plan = plan_path(target, exclude_roots=[actlize])
            plan.merge(plan_path(include))
            extend_actlize_plan(plan, include, "1.0.0")
        else:
            plan = extend_actlize_plan(plan_path(target), target, version)
        scanned, changed, renamed, _ = plan.apply(dry_run=dry_run, verbose=verbose)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"{'flex-flash-attention' if flex else 'actlize'} {version}: SDK + library compatibility")
    print(f"  Scanned: {scanned}; modified/generated: {changed}; renamed: {renamed}")
    if dry_run:
        print("  [DRY RUN - no files modified]")
    return 0


def run_flex_flash_attention(args):
    target = _require_target(args.target, "flex-flash-attention root directory")
    return _run_compatible(target, args.version, args.dry_run, args.verbose, flex=True)


def run_deepgemm(args: argparse.Namespace) -> int:
    target = _require_target(args.target, "target DeepGEMM directory")
    return _run_general(target, args.dry_run, args.verbose, label="deepgemm")


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


def run_flashmla(args: argparse.Namespace) -> int:
    target = _require_target(args.target, "target repository directory")
    if (not os.path.isfile(os.path.join(target, "setup.py")) or
            not os.path.isdir(os.path.join(target, "csrc"))):
        raise SystemExit(f"ERROR: {target} is not a FlashMLA root directory.")
    return _run_general(target, args.dry_run, args.verbose, label="flashmla")


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

    deepgemm = subparsers.add_parser(
        "deepgemm", help="Convert a DeepGEMM repository (general rules only)")
    add_common_repo_args(deepgemm)
    deepgemm.set_defaults(func=run_deepgemm)

    flashmla = subparsers.add_parser(
        "flashmla", help="Convert a FlashMLA repository (general rules only)")
    add_common_repo_args(flashmla)
    flashmla.set_defaults(func=run_flashmla)

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


def run_actlize_compat(args: argparse.Namespace) -> int:
    """Handle explicit library compatibility without changing run_actlize()."""
    target = _require_target(args.target, "actlize include directory")
    if not os.path.isdir(target):
        raise SystemExit(f"ERROR: directory not found: {target}")
    if args.version == "0.5.0":
        print("WARNING: ACTLIZE 0.5.0 is SDK-only; no validated library compatibility profile is available.")
        return run_actlize(args)
    return _run_compatible(target, args.version, args.dry_run, args.verbose)


def build_compat_parser() -> argparse.ArgumentParser:
    """Parse new compatibility commands separately from the legacy parser."""
    parser = argparse.ArgumentParser(
        description="Run opt-in ACTLIZE library compatibility conversion.")
    subparsers = parser.add_subparsers(dest="command")
    actlize = subparsers.add_parser("actlize", help="Opt into ACTLIZE library compatibility")
    actlize.add_argument("--compat", action="store_true", required=True,
                        help="Enable versioned library compatibility artifacts")
    actlize.add_argument("--version", choices=["0.5.0", "0.8.0", "1.0.0"], default="1.0.0",
                        help="Library profile version; 0.5.0 remains SDK-only")
    add_common_repo_args(actlize)
    actlize.set_defaults(func=run_actlize_compat)
    flex = subparsers.add_parser("flex-flash-attention",
                                help="Convert flex and repository-local ACTLIZE 1.0")
    flex.add_argument("--version", choices=["2.8.2"], default="2.8.2",
                      help="Supported flex version")
    add_common_repo_args(flex)
    flex.set_defaults(func=run_flex_flash_attention)
    return parser


def main_with_compat() -> int:
    """Dispatch only new commands; all legacy invocations use main() unchanged."""
    argv = sys.argv[1:]
    options = argv[1:]
    if "--" in options:
        options = options[:options.index("--")]
    if argv and (argv[0] == "flex-flash-attention"
                 or (argv[0] == "actlize" and "--compat" in options)):
        args = build_compat_parser().parse_args(argv)
        return args.func(args)
    return main()


if __name__ == "__main__":
    sys.exit(main_with_compat())
