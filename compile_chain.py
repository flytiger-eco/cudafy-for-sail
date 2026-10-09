"""General compile-chain translation rules (hgcc -> nvcc).

These rules run on every text file the general engine processes. They cover
the build-chain differences shared by all upper repositories:

  - compiler identity: hgcc -> nvcc (identifiers, paths, dict keys, labels)
  - arch flags: -arch=ppu_10 -> compute_80a/sm_80a and
    -arch=ppu_15 -> compute_89/sm_89, preserving both flags when present
  - fatbin artifact naming: hgbin -> cubin (flag and file suffix)
  - PPU-only -D defines that no converted source consumes
  - -x hg language override and the PPU LLVM backend flag prefix

Repo-specific flag-set surgery (e.g. DeepGEMM's JIT flag translation) stays
in the per-repo residual scripts.
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
    ("hgobjdump", "cuobjdump"),
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

# -x hg language override: the whole extend() line (python) or the in-string
# flag; removing only the .extend(...) call would leave a stray receiver.
DROP_XHG_LIST_RE = re.compile(r"(?m)^[ \t]*\w+\.extend\(\['-x', 'hg'\]\)\r?\n")
DROP_XHG_TEXT_RE = re.compile(r"[ ]+-x hg\b")

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
    for old, new in ARCH_FLAG_SUBS + FLAG_SUBS:
        if old in content:
            content = content.replace(old, new)

    content = DROP_DEFINE_ITEM_RE.sub("", content)
    content = DROP_DEFINE_TEXT_RE.sub("", content)
    content = DROP_XHG_LIST_RE.sub("", content)
    content = DROP_XHG_TEXT_RE.sub("", content)
    content = DROP_PPU_COMPAT_I_RE.sub("", content)
    content = DROP_COMPAT_WRAPPER_RE.sub("", content)

    for old, new in COMPILER_IDENTITY_SUBS:
        if old in content:
            content = content.replace(old, new)

    return content
