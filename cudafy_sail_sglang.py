#!/usr/bin/env python3
"""将 SGLang AOT setup 文件转换为 CUTLASS / CUDA 兼容编译配置。

默认参考 sglang-flytiger 的 setup_sail.py：替换 actlize 依赖，恢复 NVCC
编译参数与 CUDA 链接库，移除 SGL_KERNEL_SAIL 宏及本地 sailify staging。
转换后的文件直接编译源码，无需设置 SGL_KERNEL_SKIP_SAILIFY。
源码列表、绑定和 BF16/FP8 开关保持不变，源码路径默认相对化。
CUTLASS commit 从目标文件同级 CMakeLists.txt 读取，或显式指定。

支持文件路径、含 setup_sail.py 的目录及多个目标；重复运行不产生重复配置。
--no-cuda-config 可保留旧版仅替换依赖的行为。

用法：
    python cudafy_sail_sglang.py --file /path/to/setup_sail.py --dry-run
    python cudafy_sail_sglang.py /path/to/aot
    python cudafy_sail_sglang.py /path/to/setup_sail.py --out /path/to/copy.py
    python cudafy_sail_sglang.py /path/to/setup_sail.py --revert

原地转换默认保留 <文件>.actlize.bak，不覆盖已有备份。
--out 只复制文件内容：构建仍依赖目标文件同级的 csrc/include/pyproject 文件。
"""

import argparse
import ast
import difflib
import py_compile
import re
import shutil
import sys
import tempfile
from pathlib import Path

DEFAULT_FILE = (
    Path(__file__).resolve().parent.parent
    / "sglang-for-sail"
    / "python"
    / "sglang"
    / "kernels"
    / "aot"
    / "setup_sail.py"
)

# Fallback used when CMakeLists.txt cannot be parsed.  Must stay in sync with
# the repo-cutlass pin in python/sglang/kernels/aot/CMakeLists.txt.
FALLBACK_CUTLASS_COMMIT = "57e3cfb47a2d9e0d46eb6335c3dc411498efa198"
DEFAULT_CUTLASS_REPO = "https://github.com/NVIDIA/cutlass"

# Old name -> new name.  Applied case-insensitively with case preserved, so
# ``actlize``/``ACTLIZE``/``Actlize`` become ``cutlass``/``CUTLASS``/``Cutlass``.
OLD_WORD = "actlize"
NEW_WORD = "cutlass"

REPO_BLOCK_RE = re.compile(r"^[ \t]*_(?:ACTLIZE|CUTLASS)_REPO\s*=\s*_RepoInfo\(", re.M)
CUTLASS_ARCHIVE_RE = re.compile(r"NVIDIA/cutlass/archive/([0-9a-f]{40})\.tar\.gz")

# The skip-sailify branch hands absolute sources to the extension.  setuptools
# rejects that once pyproject.toml is in play: include-package-data defaults to
# true, so egg_info copies every ext source into SOURCES.txt and build_py
# asserts each entry is relative -> "setup script specifies an absolute path".
SKIP_SOURCES_RE = re.compile(
    r"^(?P<indent>[ \t]*)staged_common_sources = \[str\(root / s\) for s in "
    r"common_sources\] \+ flashinfer_sources[ \t]*$",
    re.M,
)
SKIP_SOURCES_NEW = (
    "%(indent)s# setuptools rejects absolute extension sources: with a"
    " pyproject.toml in\n"
    "%(indent)s# place include-package-data defaults to true, so egg_info copies"
    " every ext\n"
    "%(indent)s# source into SOURCES.txt and build_py then fails with 'setup"
    " script specifies\n"
    "%(indent)s# an absolute path'.  Relative paths are safe for the compile"
    " itself because\n"
    "%(indent)s# torch's ninja writer absolutises them, and this only affects"
    " the\n"
    "%(indent)s# skip-sailify branch.\n"
    "%(indent)sstaged_common_sources = common_sources + [\n"
    "%(indent)s    os.path.relpath(p, root) for p in flashinfer_sources\n"
    "%(indent)s]"
)

# 与参考脚本保持一致；内置配置使转换工具不依赖另一份仓库检出。
CUDA_FLAGS_SOURCE = '''sgl_kernel_cuda_flags = [
    "-DNDEBUG",
    f"-DOPERATOR_NAMESPACE={operator_namespace}",
    "-O3",
    "-Xcompiler",
    "-fPIC",
    "-gencode=arch=compute_80,code=sm_80",
    "-gencode=arch=compute_89,code=sm_89",
    "-std=c++17",
    "-DFLASHINFER_ENABLE_F16",
    "-DCUTE_USE_PACKED_TUPLE=1",
    "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
    "-DCUTLASS_VERSIONS_GENERATED",
    "-DCUTLASS_TEST_LEVEL=0",
    "-DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1",
    "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    f"--threads={compile_threads}",
    "-Xcompiler=-Wno-clang-format-violations",
    "-Xcompiler=-Wno-conversion",
    "-Xcompiler=-Wno-deprecated-declarations",
    "-Xcompiler=-Wno-terminate",
    "-Xcompiler=-Wfatal-errors",
    "-Xcompiler=-ftemplate-backtrace-limit=1",
    "-Xcudafe=--diag_suppress=177",
    "-Xcudafe=--diag_suppress=2361",
    f"-Xcompiler={abi_flag}",
    abi_flag,
]

if arch == "aarch64":
    sgl_kernel_cuda_flags.append("-gencode=arch=compute_87,code=sm_87")'''
CUDA_LIBRARIES = ["c10", "cuda", "cublas", "cublasLt"]
CUDA_TOOLCHAIN_GUARD = '''from torch.utils import cpp_extension as _cuda_cpp_extension

# 检查实际扩展编译模式，不以 torch.version.sail 属性单独判断。
if getattr(_cuda_cpp_extension, "IS_PPU_EXTENSION", False):
    raise RuntimeError(
        "此文件已转换为 CUDA 兼容配置，但当前 Torch 的 CUDAExtension 仍走 "
        "hgcc / cuda-free 并会原地 sailify 源码。请在 CUDA 兼容 Torch 环境中构建。"
    )'''
RELATIVE_SOURCES = (
    "common_sources + [\n"
    "    os.path.relpath(p, root) for p in flashinfer_sources\n"
    "]"
)
STAGING_SECTION_RE = re.compile(
    r"^# =+ Sailify Source Staging =+ #\n.*?"
    r"(?=^# =+ Build Extension Modules List =+ #\n)",
    re.M | re.S,
)


def _assignment(tree, name):
    """只定位顶层赋值，避免正则跨行吞掉其他配置。"""
    matches = [
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
    ]
    if len(matches) != 1:
        raise SystemExit(f"error: 必须有且只有一个顶层 {name} 赋值")
    return matches[0]


def _replace_node(text, node, replacement):
    """以 AST 行范围替换完整语句，保留其余源码和注释。"""
    lines = text.splitlines(keepends=True)
    inserted = [replacement.rstrip() + "\n"] if replacement.strip() else []
    return "".join(lines[:node.lineno - 1] + inserted + lines[node.end_lineno:])


def _rewrite_cuda_config(text, relative_sources):
    """恢复参考 NVCC 配置，移除仅用于 cuda-free 的编译和转换路径。"""
    tree = ast.parse(text)
    # 将列表和紧邻的架构追加作为一个区域替换，避免重跑累计空行。
    flags = _assignment(tree, "sgl_kernel_cuda_flags")
    if not isinstance(flags.value, ast.List):
        raise SystemExit("error: sgl_kernel_cuda_flags 不是预期的列表")
    end = flags.end_lineno
    index = tree.body.index(flags)
    following = tree.body[index + 1] if index + 1 < len(tree.body) else None
    arch_test = ast.dump(ast.parse('arch == "aarch64"', mode="eval").body)
    if isinstance(following, ast.If) and ast.dump(following.test) == arch_test:
        end = following.end_lineno
    lines = text.splitlines(keepends=True)
    while end < len(lines) and not lines[end].strip():
        end += 1
    text = "".join(
        lines[:flags.lineno - 1] + [CUDA_FLAGS_SOURCE + "\n\n"] + lines[end:]
    )
    text = re.sub(
        r"^# =+ hgcc \(device compiler\) Flags =+ #\n(?:#.*\n)*",
        "# ======================= NVCC Flags ======================= #\n",
        text, flags=re.M,
    )
    text = _rewrite_libraries(text)
    text = re.sub(
        r"^# Libraries to link\.\n(?:#.*\n)*",
        "# CUDA 链接库；cudart 由 CUDAExtension 自动追加。\n",
        text, flags=re.M,
    )
    text = re.sub(
        r"^# Upstream \(non-sailified\) include directories\..*\n(?:#.*\n)*",
        "# CUDA 兼容编译使用未转换的依赖头文件。\n",
        text, flags=re.M,
    )
    tree = ast.parse(text)
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_make_common_ops"
    ]
    if len(functions) != 1:
        raise SystemExit("error: 找不到唯一的 _make_common_ops 函数")
    nvcc = _assignment(functions[0], "nvcc_flags")
    text = _replace_node(
        text, nvcc,
        '    nvcc_flags = sgl_kernel_cuda_flags + ["-use_fast_math"]',
    )

    section = STAGING_SECTION_RE.search(text)
    if section:
        sources = RELATIVE_SOURCES if relative_sources else (
            "[str(root / s) for s in common_sources] + flashinfer_sources"
        )
        direct = (
            "# CUDA 兼容编译直接使用源码，不执行 sailify 或注入 .ppu_compat。\n"
            f"staged_common_sources = {sources}\n"
            "include_dirs = list(base_include_dirs)\n\n\n"
        )
        text = text[:section.start()] + direct + text[section.end():]
        # 删除已失去用途的 staging 导入，不改其他仍在使用的导入。
        tree = ast.parse(text)
        used_names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        for node in reversed(tree.body):
            if isinstance(node, ast.Import) and len(node.names) == 1:
                alias = node.names[0]
                if alias.name in {"atexit", "shutil"} and (
                    alias.asname or alias.name
                ) not in used_names:
                    text = _replace_node(text, node, "")
    elif any(
        isinstance(node, ast.Name) and node.id == "skip_sailify"
        for node in ast.walk(ast.parse(text))
    ):
        raise SystemExit("error: sailify staging 区域结构不符，拒绝部分转换")
    elif relative_sources:
        # 参考脚本或已转换文件也可能需要修复第三方的绝对源码路径。
        tree = ast.parse(text)
        source_name = (
            "staged_common_sources"
            if any(isinstance(n, ast.Name) and n.id == "staged_common_sources"
                   for n in ast.walk(tree))
            else "all_common_sources"
        )
        node = _assignment(tree, source_name)
        text = _replace_node(text, node, f"{source_name} = {RELATIVE_SOURCES}")
    tree = ast.parse(text)
    if not any(
        isinstance(node, ast.ImportFrom)
        and any(alias.asname == "_cuda_cpp_extension" for alias in node.names)
        for node in tree.body
    ):
        imports = [
            node for node in tree.body
            if isinstance(node, ast.ImportFrom)
            and node.module == "torch.utils.cpp_extension"
        ]
        if len(imports) != 1:
            raise SystemExit("error: 找不到唯一的 Torch 扩展编译器导入")
        node = imports[0]
        original = ast.get_source_segment(text, node)
        text = _replace_node(text, node, original + "\n" + CUDA_TOOLCHAIN_GUARD)
    return text


def _verify_cuda_config(text):
    """检查实际配置节点，不将注释中的旧平台说明误判为有效参数。"""
    tree = ast.parse(text)
    libraries = _assignment(tree, "libraries")
    if ast.literal_eval(libraries.value) != CUDA_LIBRARIES:
        raise SystemExit("error: CUDA 链接库转换不完整")
    flags = _assignment(tree, "sgl_kernel_cuda_flags")
    required = {
        elt.value for elt in ast.parse(CUDA_FLAGS_SOURCE).body[0].value.elts
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
    }
    actual = {
        elt.value for elt in flags.value.elts
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
    }
    if not required <= actual:
        raise SystemExit("error: NVCC 编译参数转换不完整")
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "-DSGL_KERNEL_SAIL" in node.value or "-U__HGGC_NO_" in node.value:
                raise SystemExit(f"error: 残留 cuda-free 编译参数 {node.value}")
        if isinstance(node, ast.Name) and node.id in {
            "skip_sailify", "_sailify_stage", "ppu_compat_dir"
        }:
            raise SystemExit(f"error: 残留 sailify 构建路径 {node.id}")


def _case_like(word, sample):
    if sample.islower():
        return word.lower()
    if sample.isupper():
        return word.upper()
    if sample[0].isupper():
        return word.capitalize()
    return word.lower()


def _replace_word(text, old, new):
    pattern = re.compile(re.escape(old), re.I)
    return pattern.sub(lambda m: _case_like(new, m.group(0)), text)


def _find_cutlass_commit(setup_path):
    """Pin cutlass to the commit the CUDA CMake build already uses."""
    cmake = setup_path.parent / "CMakeLists.txt"
    if cmake.is_file():
        match = CUTLASS_ARCHIVE_RE.search(cmake.read_text(encoding="utf-8"))
        if match:
            return match.group(1), f"CMakeLists.txt ({cmake.name})"
    return FALLBACK_CUTLASS_COMMIT, "built-in fallback"


def _rewrite_repo_block(text, repo_url, commit):
    """只修改 ACTLIZE/CUTLASS 配置块的 URL 和 commit，不影响 FlashInfer pin。"""
    lines = text.splitlines(keepends=True)
    start = None
    for idx, line in enumerate(lines):
        if REPO_BLOCK_RE.match(line):
            start = idx
            break
    if start is None:
        raise SystemExit("error: 找不到 ACTLIZE/CUTLASS 的 _RepoInfo 配置块")

    end = start
    while end < len(lines) and not lines[end].rstrip().endswith(")"):
        end += 1
    if end == len(lines):
        raise SystemExit("error: unterminated _ACTLIZE_REPO block")

    seen_repo = seen_tag = False
    for idx in range(start, end + 1):
        if re.match(r"^\s*git_repository\s*=", lines[idx]):
            newline = '    git_repository="%s",\n' % repo_url
            seen_repo = True
        elif re.match(r"^\s*git_tag\s*=", lines[idx]):
            newline = '    git_tag="%s",\n' % commit
            seen_tag = True
        else:
            continue
        indent = re.match(r"^\s*", lines[idx]).group(0)
        lines[idx] = indent + newline.lstrip()

    if not (seen_repo and seen_tag):
        raise SystemExit("error: git_repository/git_tag missing in _ACTLIZE_REPO")
    return "".join(lines)


def _rewrite_libraries(text):
    """使用参考 CUDA 库，cudart 由 CUDAExtension 自动追加。"""
    node = _assignment(ast.parse(text), "libraries")
    return _replace_node(text, node, f"libraries = {CUDA_LIBRARIES!r}")


def _rewrite_skip_sources(text):
    """Make the skip-sailify extension sources relative (setuptools requires it).

    Returns the text and whether the branch was found.
    """
    match = SKIP_SOURCES_RE.search(text)
    if not match:
        return text, False
    new = SKIP_SOURCES_NEW % {"indent": match.group("indent")}
    return text[: match.start()] + new + text[match.end() :], True


def _verify(text, repo_url=None, commit=None):
    problems = []
    if re.search(OLD_WORD, text, re.I):
        problems.append(f"residual '{OLD_WORD}' reference remains")
    block = re.search(
        r"_CUTLASS_REPO\s*=\s*_RepoInfo\((?:.|\n)*?\)", text
    )
    if not block:
        problems.append("_CUTLASS_REPO block missing after rewrite")
    else:
        body = block.group(0)
        if repo_url is not None and f'git_repository="{repo_url}"' not in body:
            problems.append("git_repository was not rewritten")
        if commit is not None and f'git_tag="{commit}"' not in body:
            problems.append(f"git_tag is not {commit}")
    if "_CUTLASS_DIR" not in text:
        problems.append("_CUTLASS_DIR not present")
    if problems:
        raise SystemExit("error: verification failed:\n  - " + "\n  - ".join(problems))


def _check_syntax(text):
    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp) / "setup_sail_cutlass.py"
        probe.write_text(text, encoding="utf-8")
        try:
            py_compile.compile(str(probe), doraise=True, cfile=str(probe) + "c")
        except py_compile.PyCompileError as exc:
            raise SystemExit(f"error: rewritten file does not compile:\n{exc}")


def _resolve_targets(raw_paths):
    """Expand CLI paths into setup files; a directory means its setup_sail.py."""
    targets = []
    for raw in raw_paths:
        path = Path(raw).expanduser().resolve()
        if path.is_dir():
            path = path / "setup_sail.py"
        if not path.is_file():
            raise SystemExit(f"error: {path} is not a file (pass a setup_sail.py)")
        if path not in targets:
            targets.append(path)
    return targets


def _convert_one(target, args):
    """逐文件转换；已替换为 CUTLASS 的文件也可补跑 CUDA 配置修复。"""
    backup = target.with_name(target.name + ".actlize.bak")
    original = target.read_text(encoding="utf-8")
    has_actlize = bool(re.search(OLD_WORD, original, re.I))

    text = original
    commit = None
    update_repo = (
        has_actlize or args.cutlass_commit
        or args.cutlass_repo != DEFAULT_CUTLASS_REPO
    )
    if update_repo:
        if args.cutlass_commit:
            commit, commit_source = args.cutlass_commit, "--cutlass-commit"
        else:
            commit, commit_source = _find_cutlass_commit(target)
        print(f"{target}: cutlass pin {commit} (from {commit_source})")
        text = _rewrite_repo_block(text, args.cutlass_repo, commit)
        text = _replace_word(text, OLD_WORD, NEW_WORD)

    _verify(text, args.cutlass_repo if update_repo else None, commit)
    if args.cuda_config:
        text = _rewrite_cuda_config(text, args.relpath_sources)
        _verify_cuda_config(text)
    else:
        if args.cuda_link_libs:
            text = _rewrite_libraries(text)
        if args.relpath_sources:
            text, _ = _rewrite_skip_sources(text)
    _check_syntax(text)

    if text == original and not args.out:
        print(f"{target}: nothing to change, already up to date")
        return
    diff = list(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            text.splitlines(keepends=True),
            fromfile=str(target),
            tofile=str(args.out or target),
            n=1,
        )
    )
    changed = sum(
        1 for line in diff if line[:1] in "+-" and line[:3] not in ("+++", "---")
    )
    print("".join(diff))
    print(f"{target}: {changed} changed line(s)")

    if args.dry_run:
        print(f"{target}: dry run, nothing written")
        return

    if args.out:
        args.out.expanduser().write_text(text, encoding="utf-8")
        print(f"wrote {args.out} (original left untouched)")
    else:
        if not args.no_backup:
            if backup.exists():
                print(f"backup: keeping existing {backup}")
            else:
                shutil.copy2(target, backup)
                print(f"backup: {backup}")
        target.write_text(text, encoding="utf-8")
        print(f"rewrote {target}")

    if args.cuda_config:
        print("CUDA 配置已转换：NVCC 参数、CUDA 库、直接源码编译，无需 skip-sailify 开关。")
    else:
        print("仅转换依赖；编译参数和 sailify 配置未转换。")


def main():
    parser = argparse.ArgumentParser(
        description="将 SGLang AOT setup 文件转换为 CUTLASS / CUDA 兼容编译配置。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "paths",
        nargs="*",
        metavar="PATH",
        help="setup_sail.py to convert, or a directory holding it (repeatable); "
        f"defaults to {DEFAULT_FILE}",
    )
    parser.add_argument(
        "-f",
        "--file",
        dest="extra_files",
        action="append",
        type=Path,
        default=[],
        help="same as PATH, kept for convenience",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        help="write the converted file here instead of editing in place",
    )
    parser.add_argument(
        "--cutlass-repo",
        default=DEFAULT_CUTLASS_REPO,
        help=f"cutlass git repository (default: {DEFAULT_CUTLASS_REPO})",
    )
    parser.add_argument(
        "--cutlass-commit",
        help="cutlass commit to pin (default: the CMakeLists.txt repo-cutlass pin)",
    )
    parser.add_argument(
        "--cuda-config",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="转换完整 CUDA 编译配置（默认开启）；--no-cuda-config 仅替换依赖",
    )
    parser.add_argument(
        "--relative-sources",
        dest="relpath_sources",
        action="store_true",
        default=True,
        help="将编译源码路径相对化，满足 setuptools 打包校验（默认开启）",
    )
    parser.add_argument(
        "--no-relative-sources",
        dest="relpath_sources",
        action="store_false",
        help="不修改源码路径；绝对路径仍可能导致 setuptools 打包失败",
    )
    parser.add_argument(
        "--cuda-link-libs",
        action="store_true",
        help="单独转换 CUDA 链接库；默认 --cuda-config 已包含此操作",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="show the diff, write nothing"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="do not keep <file>.actlize.bak before an in-place rewrite",
    )
    parser.add_argument(
        "--revert", action="store_true", help="restore <file>.actlize.bak"
    )
    args = parser.parse_args()

    raw = list(args.paths) + [str(path) for path in args.extra_files]
    if not raw:
        if not DEFAULT_FILE.is_file():
            raise SystemExit(
                "error: no PATH given and the default AOT setup file is missing; "
                "pass one explicitly"
            )
        raw = [str(DEFAULT_FILE)]
        print(f"no PATH given, using default {DEFAULT_FILE}")

    targets = _resolve_targets(raw)
    if args.out and len(targets) > 1:
        raise SystemExit("error: --out accepts a single PATH")
    if args.out and args.revert:
        raise SystemExit("error: --out cannot be combined with --revert")
    if args.out and args.out.expanduser().resolve() == targets[0]:
        args.out = None  # 相同路径仍按原地转换处理，以保留备份。

    failed = 0
    for target in targets:
        if args.revert:
            backup = target.with_name(target.name + ".actlize.bak")
            if not backup.is_file():
                print(f"error: {target}: no backup at {backup}", file=sys.stderr)
                failed += 1
                continue
            if args.dry_run:
                print(f"dry run: 将从 {backup} 恢复 {target}，未写入")
            else:
                shutil.copy2(backup, target)
                print(f"restored {target} from {backup}")
            continue
        try:
            _convert_one(target, args)
        except (SystemExit, OSError, SyntaxError, ValueError) as exc:
            message = str(exc)
            if not message.startswith("error:"):
                message = f"error: {target}: {message}"
            print(message, file=sys.stderr)
            failed += 1

    if failed:
        print(f"\n{failed} of {len(targets)} target(s) failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

