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

# PPU-toolchain option values that must keep their original spelling: they
# are consumed by the PPU-side nvcc wrapper, whose accepted option names are
# hggc-spelled ('-no-hggc-embed-bc' is recognized, '-no-cuda-embed-bc' is
# not).
_PROTECTED_STRINGS = [
    "--no-hggc-embed-bc",
    "-no-hggc-embed-bc",
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


def _iter_files(root: str):
    for base, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames
            if d not in SKIP_DIRS and not d.endswith(SKIP_DIR_SUFFIXES)
        )
        for fname in sorted(filenames):
            filepath = os.path.join(base, fname)
            if _is_target(filepath):
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


def convert_file(filepath: str, dry_run: bool = False) -> bool:
    """Convert one file; returns True only after a successful write."""
    try:
        original = _read(filepath)
    except OSError as exc:
        print(f"  [general] WARNING: cannot read {filepath}: {exc}")
        return False
    converted = convert_text(original)
    if converted == original:
        return False
    if not dry_run:
        try:
            _write(filepath, converted)
        except OSError as exc:
            print(f"  [general] WARNING: cannot write {filepath}: {exc}")
            return False
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


def convert_path(target: str, dry_run: bool = False, verbose: bool = False):
    """Convert a file or a whole directory tree.

    Content conversion runs first; afterwards, files and directories whose
    names the map converts are renamed so that rewritten includes keep
    resolving. Returns (scanned, changed, renamed, changed_paths).
    """
    if os.path.isfile(target):
        changed = convert_file(target, dry_run)
        return 1, (1 if changed else 0), 0, ([target] if changed else [])

    roots = [os.path.abspath(target)] + _extra_roots(target)
    scanned = 0
    changed_paths = []
    for root in roots:
        for filepath in _iter_files(root):
            scanned += 1
            if convert_file(filepath, dry_run):
                changed_paths.append(filepath)
                if verbose:
                    print(f"  [general] {os.path.relpath(filepath, target)}")

    renames = rename_path(target, dry_run=dry_run, verbose=verbose)
    return scanned, len(changed_paths), len(renames), changed_paths


# ---------------------------------------------------------------------------
# File / directory renames
# ---------------------------------------------------------------------------

def _convert_name(name: str) -> str:
    """Convert a single path component with the same rules as file content."""
    return convert_text(name)


def _plan_renames(root: str):
    """Collect (old, new) rename pairs for convertible files and directories."""
    renames = []
    for base, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames
            if d not in SKIP_DIRS and not d.endswith(SKIP_DIR_SUFFIXES)
        )
        for fname in filenames:
            filepath = os.path.join(base, fname)
            if not _is_target(filepath):
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
            print(f"  [general] WARNING: rename target already exists, skipped: "
                  f"{old} -> {new}")
            continue
        if dry_run:
            print(f"  [general] would rename: {old} -> {new}")
            done.append((old, new))
            continue
        try:
            os.rename(old, new)
        except OSError as exc:
            print(f"  [general] WARNING: cannot rename {old} -> {new}: {exc}")
            continue
        if verbose:
            print(f"  [general] renamed: {old} -> {new}")
        done.append((old, new))
    return done


def rename_path(target: str, dry_run: bool = False, verbose: bool = False):
    """Rename convertible files/directories under target.

    Returns the list of applied (or, on dry-run, planned) renames.
    """
    if not os.path.isdir(target):
        return []
    renames = _plan_renames(os.path.abspath(target))
    for extra in _extra_roots(target):
        renames.extend(_plan_renames(extra))
    return _apply_renames(renames, dry_run=dry_run, verbose=verbose)


def _walk_tree(root, exclude_roots=()):
    """Strict traversal reserved for opt-in compatibility plans."""
    def onerror(exc):
        raise exc
    for base, dirnames, filenames in os.walk(root, onerror=onerror):
        dirnames[:] = sorted(
            d for d in dirnames
            if d not in SKIP_DIRS and not d.endswith(SKIP_DIR_SUFFIXES)
            and not any(_within(os.path.join(base, d), skip) for skip in exclude_roots)
        )
        for name in dirnames:
            path = os.path.join(base, name)
            if os.path.islink(path):
                raise ValueError(f"Conversion directory contains a symlink: {path}")
        yield base, dirnames, filenames


def _iter_plan_files(root, exclude_roots=()):
    for base, _, filenames in _walk_tree(root, exclude_roots):
        for name in sorted(filenames):
            path = os.path.join(base, name)
            if _is_target(path):
                yield path


def _plan_strict_renames(root, exclude_roots=()):
    renames = []
    for base, dirnames, filenames in _walk_tree(root, exclude_roots):
        for name in filenames:
            path = os.path.join(base, name)
            converted = _convert_name(name)
            if _is_target(path) and converted != name:
                renames.append((path, os.path.join(base, converted)))
        for name in dirnames:
            converted = _convert_name(name)
            if converted != name:
                renames.append((os.path.join(base, name), os.path.join(base, converted)))
    return renames


def _validate_rename_destinations(renames):
    """Reject destinations reserved by distinct sources before any writes."""
    destinations = {}
    for old, new in renames:
        old, new = os.path.abspath(old), os.path.abspath(new)
        previous = destinations.setdefault(new, old)
        if previous != old:
            raise ValueError(f"Duplicate rename destination: {previous}, {old} -> {new}")


class ConversionPlan:
    """Shared SDK/library virtual tree; planning never writes to disk."""

    def __init__(self):
        self.files = {}
        self.sources = {}
        self.originals = {}
        self.renames = []
        self.diagnostics = []
        self.scanned = 0

    def merge(self, other):
        renames = self.renames + [pair for pair in other.renames if pair not in self.renames]
        _validate_rename_destinations(renames)
        for path, text in other.files.items():
            if path in self.files and (self.files[path] != text
                                       or self.sources[path] != other.sources[path]):
                raise ValueError(f"Conversion plan path conflict: {path}")
            self.files[path] = text
            self.sources[path] = other.sources[path]
        self.originals.update(other.originals)
        self.renames = renames
        self.scanned = len(self.originals)
        self.diagnostics.extend(other.diagnostics)
        return self

    def _physical_path(self, path):
        # Check physical occupancy before a containing directory is renamed.
        for old, new in sorted(self.renames, key=lambda pair: pair[0].count(os.sep)):
            if os.path.isdir(old) and _within(path, new):
                path = os.path.join(old, os.path.relpath(path, new))
        return path

    def _check_generated_path(self, path):
        physical = self._physical_path(path)
        if os.path.realpath(physical) != physical:
            raise ValueError(f"Generated path contains a symlink: {path}")
        if os.path.lexists(physical):
            raise ValueError(f"Generated path is occupied by a file or directory: {path}")
        parent = os.path.dirname(physical)
        while not os.path.lexists(parent):
            parent = os.path.dirname(parent)
        if not os.path.isdir(parent):
            raise ValueError(f"Generated path parent is not a directory: {path}")

    def add_file(self, path, text):
        path = os.path.abspath(path)
        if path in self.files:
            if self.files[path] != text:
                raise ValueError(f"Generated rule path conflict: {path}")
            return
        self._check_generated_path(path)
        if any(_within(path, existing) or _within(existing, path) for existing in self.files):
            raise ValueError(f"Generated path conflicts with the virtual file hierarchy: {path}")
        self.files[path] = text
        self.sources[path] = None

    def changed_paths(self):
        return [path for path, text in self.files.items()
                if self.sources[path] is None
                or text != self.originals[self.sources[path]]]

    def apply(self, dry_run=False, verbose=False):
        _validate_rename_destinations(self.renames)
        changed = self.changed_paths()
        if verbose or dry_run:
            for path in changed:
                print(f"  [plan] {'generate' if self.sources[path] is None else 'modify'}: {path}")
            for old, new in self.renames:
                print(f"  [plan] rename: {old} -> {new}")
        for note in self.diagnostics:
            print(f"  [compat] {note}")
        if dry_run:
            return self.scanned, len(changed), len(self.renames), changed
        # Verify source files have not changed before the first write.
        for source, original in self.originals.items():
            if os.path.realpath(source) != source:
                raise ValueError(f"Conversion file contains a symlink: {source}")
            if _read(source) != original:
                raise ValueError(f"Source file changed after planning: {source}")
        for old, new in self.renames:
            if os.path.lexists(new):
                raise ValueError(f"Rename target conflict: {old} -> {new}")
        for path in changed:
            if self.sources[path] is None:
                self._check_generated_path(path)
        for path in changed:
            source = self.sources[path]
            if source is not None:
                _write(source, self.files[path])
        for old, new in sorted(self.renames, key=lambda pair: -pair[0].count(os.sep)):
            os.rename(old, new)
        for path in changed:
            if self.sources[path] is None:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                _write(path, self.files[path])
        return self.scanned, len(changed), len(self.renames), changed


def _within(path, root):
    return os.path.commonpath([os.path.abspath(path), os.path.abspath(root)]) == os.path.abspath(root)


def plan_path(target, exclude_roots=()):
    """Build a strict, read-only SDK plan for opt-in library processing."""
    target = os.path.abspath(target)
    if not os.path.exists(target):
        raise ValueError(f"Conversion target does not exist: {target}")
    excluded = [os.path.abspath(root) for root in exclude_roots]
    plan = ConversionPlan()
    roots = [target] + (_extra_roots(target) if os.path.isdir(target) else [])
    for root in roots:
        if os.path.realpath(root) != root:
            raise ValueError(f"Conversion root contains a symlink: {root}")
        paths = [root] if os.path.isfile(root) else _iter_plan_files(root, excluded)
        for source in paths:
            if any(_within(source, skip) for skip in excluded):
                continue
            if os.path.realpath(source) != source:
                raise ValueError(f"Conversion file contains a symlink: {source}")
            relative = os.path.relpath(source, root)
            final = source if os.path.isfile(root) else os.path.join(
                root, *[_convert_name(part) for part in relative.split(os.sep)])
            if final in plan.files:
                raise ValueError(f"Multiple source files map to the same path: {final}")
            original = _read(source)
            plan.files[final] = convert_text(original)
            plan.sources[final] = source
            plan.originals[source] = original
        if os.path.isdir(root):
            for old, new in _plan_strict_renames(root, excluded):
                if any(_within(old, skip) for skip in excluded):
                    continue
                if os.path.lexists(new):
                    raise ValueError(f"Rename target already exists: {old} -> {new}")
                plan.renames.append((old, new))
    _validate_rename_destinations(plan.renames)
    plan.scanned = len(plan.originals)
    return plan
