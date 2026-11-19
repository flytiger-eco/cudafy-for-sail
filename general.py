"""General conversion engine: full SDK map + compile-chain rules.

convert_path() walks a target tree and rewrites every text source file with
four passes:

  1. compile chain (compile_chain.py) -- hgcc -> nvcc, arch flags, PPU-only
     defines; runs first so its drop rules see the original flag text;
  2. string keys -- map entries that are not plain identifiers (diagnostic
     phrases, header names, include-path fragments, sonames);
  3. identifiers -- exact token lookup plus an underscore-boundary suffix
     lookup (lazy_hgModuleLoad, DG_JIT_USE_HGRTC, ...);
  4. prefix-open keys -- entries that deliberately match camelCase
     continuations (acblasSgemmBatched, __HGGC_STD_*).

`#pragma hggc` lines and PPU-toolchain option values are stashed away first
and restored verbatim: the cuda-compatible nvcc still honors them under
their original spelling. All passes are idempotent.

After the content pass, files and directories whose names the map converts
are renamed so rewritten includes keep resolving.
"""

import os
import re
import shutil
import tempfile

from compile_chain import apply_compile_chain
from sdk_map import (
    NON_SDK_ARTIFACT_MAP,
    NON_SDK_COMPILER_MACROS,
    NON_SDK_PREFIX_OPEN,
    NON_SDK_SAILIFY_MAP,
    NON_SDK_TEXT_MAP,
    NON_SDK_TOKEN_MAP,
    SDK_DIRECTORY_MAP,
    SDK_FUNCTION_MAP,
    SDK_HEADER_MAP,
    SDK_MACRO_MAP,
    SDK_PREFIX_OPEN,
    SDK_TYPE_MAP,
)

# Combined exact-token lookup: every categorized map merged. The categories
# exist for readability in sdk_map.py; conversion semantics are identical.
SDK_TOKEN_MAP = {}
for _map in (SDK_DIRECTORY_MAP, SDK_HEADER_MAP, SDK_FUNCTION_MAP,
             SDK_TYPE_MAP, SDK_MACRO_MAP, NON_SDK_COMPILER_MACROS,
             NON_SDK_ARTIFACT_MAP, NON_SDK_TEXT_MAP, NON_SDK_TOKEN_MAP,
             NON_SDK_SAILIFY_MAP):
    SDK_TOKEN_MAP.update(_map)

# ---------------------------------------------------------------------------
# Pass construction (done once at import time)
# ---------------------------------------------------------------------------

_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_IS_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")

# Entries whose key is not a plain identifier.
_STRING_KEYS = {
    k: v for k, v in SDK_TOKEN_MAP.items() if not _IS_IDENTIFIER.match(k)
}
_STRING_PATTERN = re.compile(
    "|".join(re.escape(k) for k in sorted(_STRING_KEYS, key=len, reverse=True))
)

# Prefix-open entries: map keys ending with '_' plus the curated camelCase
# prefixes from the generated map data.
_PREFIX_OPEN = sorted(
    ((k, v) for k, v in SDK_TOKEN_MAP.items() if k.endswith("_")),
    key=lambda kv: -len(kv[0]),
)
_PREFIX_OPEN += sorted(
    list(SDK_PREFIX_OPEN.items()) + list(NON_SDK_PREFIX_OPEN.items()),
    key=lambda kv: -len(kv[0]),
)

# `#pragma hggc ...` lines stay EXACTLY as they are: the cuda-compat nvcc
# still honors the hggc pragma namespace, so rewriting the namespace would
# silently drop the pragma's effect (e.g. FA3's `#pragma hggc
# mmatiestrictly`). The marker is digits + NUL only, which no pass touches.
_PRAGMA_LINE_RE = re.compile(r"(?m)^\s*#\s*pragma\s+hggc\b[^\n]*$")

# Toolchain option values, archive names and environment variables that keep
# their spelling. The PPU-side nvcc wrapper recognizes '-no-hggc-embed-bc',
# not '-no-cuda-embed-bc'.
_PROTECTED_STRINGS = [
    "--no-hggc-embed-bc",
    "-no-hggc-embed-bc",
    "sailSHMEM_hggcrt",
    "DEEPEP_DISABLE_HGTX",
]


def _protect_pragma_lines(content: str):
    stash = []

    def _repl(match):
        stash.append(match.group(0))
        return "\x00%d\x00" % (len(stash) - 1)

    return _PRAGMA_LINE_RE.sub(_repl, content), stash


def _restore_pragma_lines(content: str, stash) -> str:
    for idx, line in enumerate(stash):
        content = content.replace("\x00%d\x00" % idx, line)
    return content


def _protect_strings(content: str):
    stash = []
    for s in _PROTECTED_STRINGS:
        while s in content:
            content = content.replace(s, "\x01%d\x01" % len(stash), 1)
            stash.append(s)
    return content, stash


def _restore_strings(content: str, stash) -> str:
    for idx, s in enumerate(stash):
        content = content.replace("\x01%d\x01" % idx, s)
    return content


def _identifier_sub(match):
    token = match.group(0)
    sub = SDK_TOKEN_MAP.get(token)
    if sub is not None:
        return sub
    # Underscore-boundary suffix: the longest suffix of the token that starts
    # right after one of the token's '_' characters and is itself a key
    # (lazy_hgModuleLoad -> lazy_cuModuleLoad, DG_JIT_USE_HGRTC -> ...NVRTC).
    pos = token.find("_")
    while pos != -1 and pos + 1 < len(token):
        suffix = token[pos + 1:]
        sub = SDK_TOKEN_MAP.get(suffix)
        if sub is not None:
            return token[: pos + 1] + sub
        pos = token.find("_", pos + 1)
    return token


def convert_text(content: str) -> str:
    """Apply every general rule to one file's content."""
    content, pragma_stash = _protect_pragma_lines(content)
    content, string_stash = _protect_strings(content)
    content = apply_compile_chain(content)
    if _STRING_PATTERN.pattern:
        content = _STRING_PATTERN.sub(
            lambda m: _STRING_KEYS[m.group(0)], content
        )
    content = _IDENTIFIER_RE.sub(_identifier_sub, content)
    for old, new in _PREFIX_OPEN:
        if old in content:
            content = content.replace(old, new)
    content = _restore_strings(content, string_stash)
    return _restore_pragma_lines(content, pragma_stash)


# ---------------------------------------------------------------------------
# Tree walking
# ---------------------------------------------------------------------------

TEXT_EXTENSIONS = {
    ".h", ".hpp", ".cuh", ".cu", ".c", ".cpp", ".cc", ".cxx", ".inl", ".inc",
    ".py", ".sh", ".cmake", ".md", ".jinja",
}
TEXT_FILENAMES = {"CMakeLists.txt", "Makefile"}

SKIP_DIRS = {".git", "__pycache__", "build", "dist", ".eggs", "node_modules"}
SKIP_DIR_SUFFIXES = (".egg-info", ".dist-info")


def _is_target(filepath: str) -> bool:
    name = os.path.basename(filepath)
    if name in TEXT_FILENAMES:
        return True
    return os.path.splitext(name)[1] in TEXT_EXTENSIONS


def _excluded(path: str, root: str, exclude) -> bool:
    relative = os.path.relpath(path, root)
    return any(relative == item or relative.startswith(item.rstrip(os.sep) + os.sep)
               for item in exclude)


def _iter_files(root: str, exclude=()):
    for base, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames
            if d not in SKIP_DIRS and not d.endswith(SKIP_DIR_SUFFIXES)
            and not _excluded(os.path.join(base, d), root, exclude)
            and not os.path.islink(os.path.join(base, d))
        )
        for fname in sorted(filenames):
            filepath = os.path.join(base, fname)
            if (_is_target(filepath) and not _excluded(filepath, root, exclude)
                    and not os.path.islink(filepath)):
                yield filepath


def _read(filepath: str):
    # surrogateescape preserves non-UTF-8 bytes; newline="" preserves CRLF.
    with open(filepath, "r", encoding="utf-8", errors="surrogateescape",
              newline="") as f:
        return f.read()


def _write(filepath: str, content: str) -> None:
    with open(filepath, "w", encoding="utf-8", errors="surrogateescape",
              newline="") as f:
        f.write(content)


def convert_file(filepath: str, dry_run: bool = False,
                 file_converter=None, relative=None) -> bool:
    """Convert one file; returns True only after a successful write."""
    try:
        original = _read(filepath)
    except OSError as exc:
        print(f"  [general] WARNING: cannot read {filepath}: {exc}")
        raise
    converted = (file_converter(relative, original) if file_converter is not None
                 else convert_text(original))
    if converted == original:
        return False
    if filepath.endswith('.py'):
        try:
            compile(original, filepath, 'exec')
        except SyntaxError:
            pass
        else:
            compile(converted, filepath, 'exec')
    if not dry_run:
        try:
            _write(filepath, converted)
        except OSError as exc:
            print(f"  [general] WARNING: cannot write {filepath}: {exc}")
            raise
    return True


def _extra_roots(root: str):
    """Sibling trees that belong to the target.

    An actlize-style include directory (basenamed 'include') carries its
    tools/util/include sibling, which is always converted together with it.
    """
    root = os.path.abspath(root)
    extras = []
    if os.path.basename(root) == "include":
        tools_util = os.path.normpath(
            os.path.join(root, "..", "tools", "util", "include")
        )
        if os.path.isdir(tools_util):
            extras.append(tools_util)
    return extras


def convert_path(target: str, dry_run: bool = False, verbose: bool = False,
                 output=None, exclude=(), file_converter=None):
    """Convert a file or a whole directory tree.

    Content conversion runs first; afterwards, files and directories whose
    names the map converts are renamed so that rewritten includes keep
    resolving. Returns (scanned, changed, renamed, changed_paths).
    """
    if output is not None:
        source, destination = os.path.abspath(target), os.path.abspath(output)
        if not os.path.isdir(source):
            raise ValueError('--output requires a source directory')
        if os.path.commonpath([source, destination]) == source:
            raise ValueError('Output directory must be outside the source directory')
        if os.path.lexists(destination):
            raise FileExistsError(destination)
        if dry_run:
            print(f'  [general] would copy: {source} -> {destination}')
            return convert_path(source, True, verbose, exclude=exclude,
                                file_converter=file_converter)
        parent = os.path.dirname(destination)
        os.makedirs(parent, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.cudafy-', dir=parent) as temporary:
            staged = os.path.join(temporary, os.path.basename(source))
            shutil.copytree(source, staged, symlinks=True)
            result = convert_path(staged, verbose=verbose, exclude=exclude,
                                  file_converter=file_converter)
            os.rename(staged, destination)
        return result
    if os.path.isfile(target):
        changed = convert_file(target, dry_run, file_converter, os.path.basename(target))
        return 1, (1 if changed else 0), 0, ([target] if changed else [])

    roots = [os.path.abspath(target)] + _extra_roots(target)
    scanned = 0
    changed_paths = []
    for root in roots:
        for filepath in _iter_files(root, exclude):
            scanned += 1
            relative = os.path.relpath(filepath, target).replace(os.sep, '/')
            if convert_file(filepath, dry_run, file_converter, relative):
                changed_paths.append(filepath)
                if verbose:
                    print(f"  [general] {os.path.relpath(filepath, target)}")

    renames = rename_path(target, dry_run=dry_run, verbose=verbose, exclude=exclude)
    return scanned, len(changed_paths), len(renames), changed_paths


# ---------------------------------------------------------------------------
# File / directory renames
# ---------------------------------------------------------------------------

def _convert_name(name: str) -> str:
    """Convert a single path component with the same rules as file content."""
    return convert_text(name)


def _plan_renames(root: str, exclude=()):
    """Collect (old, new) rename pairs for convertible files and directories."""
    renames = []
    for base, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames
            if d not in SKIP_DIRS and not d.endswith(SKIP_DIR_SUFFIXES)
            and not _excluded(os.path.join(base, d), root, exclude)
            and not os.path.islink(os.path.join(base, d))
        )
        for fname in filenames:
            filepath = os.path.join(base, fname)
            if (not _is_target(filepath) or _excluded(filepath, root, exclude)
                    or os.path.islink(filepath)):
                continue
            new_name = _convert_name(fname)
            if new_name != fname:
                renames.append((filepath, os.path.join(base, new_name)))
        for dname in dirnames:
            new_name = _convert_name(dname)
            if new_name != dname:
                renames.append((os.path.join(base, dname), os.path.join(base, new_name)))
    return renames


def _apply_renames(renames, dry_run: bool = False, verbose: bool = False):
    """Apply planned renames deepest-first so parents move children along."""
    done = []
    for old, new in sorted(renames, key=lambda pair: -pair[0].count(os.sep)):
        if os.path.exists(new):
            raise FileExistsError(f"Rename target already exists: {old} -> {new}")
        if dry_run:
            print(f"  [general] would rename: {old} -> {new}")
            done.append((old, new))
            continue
        try:
            os.rename(old, new)
        except OSError as exc:
            print(f"  [general] WARNING: cannot rename {old} -> {new}: {exc}")
            raise
        if verbose:
            print(f"  [general] renamed: {old} -> {new}")
        done.append((old, new))
    return done


def rename_path(target: str, dry_run: bool = False, verbose: bool = False,
                exclude=()):
    """Rename convertible files/directories under target.

    Returns the list of applied (or, on dry-run, planned) renames.
    """
    if not os.path.isdir(target):
        return []
    renames = _plan_renames(os.path.abspath(target), exclude)
    for extra in _extra_roots(target):
        renames.extend(_plan_renames(extra, exclude))
    return _apply_renames(renames, dry_run=dry_run, verbose=verbose)
