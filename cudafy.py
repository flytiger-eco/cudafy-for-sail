#!/usr/bin/env python3
"""cudafy-for-sail 的统一 CUDA 兼容转换入口。

  python3 cudafy.py general <path>
      仅执行 SDK 命名与编译链转换，不生成 ACTLIZE 库级兼容产物。
      未知仓库名，以及 FA2/xformers/deepgemm/flashmla 入口仍使用通用规则。

  python3 cudafy.py actlize --version=0.8.0|1.0.0 <include>
      在同一只读计划中追加对应版本的库级规则；0.5.0 仅执行 SDK 转换。

  python3 cudafy.py flex-flash-attention --version=2.8.2 <root>
      转换外层 SDK 内容，并为仓内 ACTLIZE 的 include/util 生成 1.0 兼容产物。
      不自动修改 ACTLIZE_DIR 指定的外部库；dry-run 不写入任何转换产物。
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

KNOWN_COMMANDS = {
    "general", "actlize", "flash-attention", "flex-flash-attention",
    "xformers", "deepgemm", "flashmla",
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
    if args.version == "0.5.0":
        print("WARNING: ACTLIZE 0.5.0 仅执行 SDK 转换，没有经过验证的库兼容 profile。")
        return _run_general(target, args.dry_run, args.verbose,
                            version=args.version, label="actlize")
    return _run_compatible(target, args.version, args.dry_run, args.verbose)


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
                raise ValueError(f"不是受支持的 flex 根目录：{target}")
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
    print(f"{'flex-flash-attention' if flex else 'actlize'} {version}: SDK + 库级兼容")
    print(f"  扫描：{scanned}；修改/生成：{changed}；重命名：{renamed}")
    if dry_run:
        print("  [DRY RUN - no files modified]")
    return 0


def run_flex_flash_attention(args):
    target = _require_target(args.target, "flex-flash-attention 根目录")
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
        "actlize", help="转换 ACTLIZE：SDK 规则及版本化库兼容层")
    actlize.add_argument(
        "--version", choices=["0.5.0", "0.8.0", "1.0.0"], default="1.0.0",
        help="ACTLIZE 库版本；0.8/1.0 启用兼容 profile，0.5 仅执行 SDK 转换")
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

    flex = subparsers.add_parser("flex-flash-attention", help="转换 flex 及仓内 ACTLIZE 1.0")
    flex.add_argument("--version", choices=["2.8.2"], default="2.8.2",
                      help="已适配的 flex 版本")
    add_common_repo_args(flex)
    flex.set_defaults(func=run_flex_flash_attention)

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


if __name__ == "__main__":
    sys.exit(main())
