"""General compile-chain translation rules (hgcc -> nvcc).

These rules run on every text file the general engine processes. They cover
the build-chain differences shared by all upper repositories:

  - compiler identity: hgcc -> nvcc (identifiers, paths, dict keys, labels)
  - arch flags: -arch=ppu_10 -> compute_80a/sm_80a and
    -arch=ppu_15 -> compute_89/sm_89, preserving both flags when present
  - fatbin artifact naming: hgbin -> cubin (flag and file suffix)
  - PPU-only -D defines that no converted source consumes
  - -x hg -> -x cu and the PPU LLVM backend flag prefix
  - resource reporting and local-memory warnings

Build dependencies and reporting options are declared by the source
repositories; this module does not inject repository-specific build settings.
"""

import re

# ---------------------------------------------------------------------------
# Compiler identity renames (plain substring, case-aware).
# ---------------------------------------------------------------------------
COMPILER_IDENTITY_SUBS = [
    # lowercase first: covers hgcc_flags, get_hgcc_compiler, "hgcc":,
    # bin/hgcc, hgcc.tmp., append_hgcc_threads, extra_hgcc_args ...
    ("hgcc", "nvcc"),
    # uppercase: HGCC compiler: / HGCCBuildExtension-style labels
    ("HGCC", "NVCC"),
    # ("hgobjdump", "cuobjdump"),  # no cuobjdump in the CUDA-compatible SDK
]

# After conversion the SDK root variable resolves to <SDK>/CUDA_SDK, which
# ships nvcc but no objdump at all; the PPU tool still lives in <SDK>/bin, one
# level up, so anchor these paths there instead of renaming the tool.
SDK_TOOL_PATH_SUBS = [
    ('sdk_home / "bin" / "hgobjdump"', 'sdk_home / ".." / "bin" / "hgobjdump"'),
]

# ---------------------------------------------------------------------------
# Arch flags.
# ---------------------------------------------------------------------------
GENCODE_SM80a = "-gencode=arch=compute_80a,code=sm_80a"
GENCODE_SM89 = "-gencode=arch=compute_89,code=sm_89"

# Any remaining single arch flag, in any quoting style.
ARCH_FLAG_SUBS = [
    ("-arch=ppu_10", GENCODE_SM80a),
    ("-arch=ppu_15", GENCODE_SM89),
]

# ---------------------------------------------------------------------------
# Flag-level renames and removals.
# ---------------------------------------------------------------------------
FLAG_SUBS = [
    # PPU LLVM backend prefix -> nvcc's mllvm
    ("-Xllvm", "-mllvm"),
    # fatbin-producing flag
    ("-hgbin", "-cubin"),
]

_REPORT_OPTIONS = {
    '--resource-usage': '--ptxas-options=--verbose',
    '-resource-usage': '--ptxas-options=--verbose',
    '--ppu-warn-lmem-usage': '--ptxas-options=--warn-on-local-memory-usage',
    '-ppu-warn-lmem-usage': '--ptxas-options=--warn-on-local-memory-usage',
}
_LLVM_OPTION = r"(?:--llvm-options|-Xllvm)"
_REPORT_VALUE = r"--?(?:resource-usage|ppu-warn-lmem-usage)"
REPORT_FLAG_RE = re.compile(
    r"(?<![\w-])" + _LLVM_OPTION + r"(?:=|[ \t]+)(" + _REPORT_VALUE + r")(?![\w,=.-])"
)
REPORT_LIST_RE = re.compile(
    r"(?P<q>['\"])" + _LLVM_OPTION + r"(?P=q)\s*,\s*"
    r"(?P<v>['\"])(?P<report>" + _REPORT_VALUE + r")(?P=v)"
)

# PPU-only -D defines; no converted source consumes them (verified: they only
# ever appear on command lines). Removed as standalone list items ...
DROP_DEFINE_ITEM_RE = re.compile(
    r"[ \t]*[\"']-DUSE_(?:HGGC|CLANG|ACWRAPPER)(?:=[^\s\"']*)?[\"'],?"
)
# ... or as in-string flag text.
DROP_DEFINE_TEXT_RE = re.compile(
    r"(?<![A-Za-z0-9_])-DUSE_(?:HGGC|CLANG|ACWRAPPER)(?![A-Za-z0-9_])"
    r"(?:=[^\s\"']*)?"
)

# Preserve an explicit language override for inputs with non-device suffixes.
XHG_TEXT_RE = re.compile(r"(?<![\w-])(-x(?:=|[ \t]+))hg(?![\w.-])")
XHG_LIST_RE = re.compile(r"((['\"])-x\2\s*,\s*)(['\"])hg\3")

_FSTR_FIELD = r"\{[^{}\n]*\}"
_FSTR_FIELD_COMPAT = r"\{[^{}\n]*\.ppu_compat[^{}\n]*\}"
_SUBST_FIELD = r"\$\([^()\n]*\)"
_PATH_CHAR = r"[^\s\"'{}\[\]();,<>|&#`]"
_FLAG_PATH = (
    r"(?:" + _FSTR_FIELD + r"|" + _SUBST_FIELD + r"|" + _PATH_CHAR + r")*"
)
_COMPAT_I_PATH = (
    _FLAG_PATH + r"(?:\.ppu_compat|" + _FSTR_FIELD_COMPAT + r")" + _FLAG_PATH
)
_COMPAT_I_ELEMENT = r"[fF]?(?P<q>['\"])-I" + _COMPAT_I_PATH + r"(?P=q)"

# Whole-line matches only: the sole shape in the conversion targets is a list
# literal with one flag per line. Inline shapes (.append() arguments, conditional
# elements, tuples, "-include=...") are left intact -- a partial edit there yields
# SyntaxError/TypeError, while an unconverted flag merely stays visible.
DROP_PPU_COMPAT_I_RE = re.compile(
    r"^[ \t]*" + _COMPAT_I_ELEMENT + r",?[ \t]*\r?\n", re.M
)
DROP_COMPAT_WRAPPER_RE = re.compile(
    r"(?m)^[ \t]*['\"]-include['\"],[ \t]*['\"]compatible_wrapper\.h['\"],?[ \t]*\r?\n"
    r"|^[ \t]*-include[ \t=]+(['\"]?)compatible_wrapper\.h\1,?[ \t]*\r?\n"
)


def apply_compile_chain(content: str) -> str:
    """Apply the general compile-chain rules to one file's content."""
    content = REPORT_LIST_RE.sub(lambda m: m['q'] + _REPORT_OPTIONS[m['report']] + m['q'], content)
    content = REPORT_FLAG_RE.sub(lambda m: _REPORT_OPTIONS[m[1]], content)
    for old, new in ARCH_FLAG_SUBS + FLAG_SUBS:
        if old in content:
            content = content.replace(old, new)

    content = DROP_DEFINE_ITEM_RE.sub("", content)
    content = DROP_DEFINE_TEXT_RE.sub("", content)
    content = XHG_LIST_RE.sub(lambda m: m[1] + m[3] + 'cu' + m[3], content)
    content = XHG_TEXT_RE.sub(r"\1cu", content)
    content = DROP_PPU_COMPAT_I_RE.sub("", content)
    content = DROP_COMPAT_WRAPPER_RE.sub("", content)

    for old, new in COMPILER_IDENTITY_SUBS + SDK_TOOL_PATH_SUBS:
        if old in content:
            content = content.replace(old, new)

    return content
