"""General compile-chain translation rules (hgcc -> nvcc).

These rules run on every text file the general engine processes. They cover
the build-chain differences shared by all upper repositories:

  - compiler identity: hgcc -> nvcc (identifiers, paths, dict keys, labels)
  - arch flags: -arch=ppu_10 / -arch=ppu_15 -> a single
    -gencode=arch=compute_80,code=sm_80 (the sm_80 gencode compiles device
    code for both PPU generations)
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
GENCODE_SM80 = "-gencode=arch=compute_80,code=sm_80"

# The common setup.py shape: two adjacent append() calls selecting ppu_10 and
# ppu_15. Collapsed into ONE gencode appends, mirroring the FlashMLA
# conversion (any receiver variable name, either arch order).
ARCH_BLOCK_RE = re.compile(
    r"(?P<indent>[ \t]*)(?P<recv>\w+)\.append\((?P<q1>[\"'])-arch=ppu_(?:10|15)(?P=q1)\)\r?\n"
    r"[ \t]*(?P=recv)\.append\((?P<q2>[\"'])-arch=ppu_(?:10|15)(?P=q2)\)"
)

# Any remaining single arch flag, in any quoting style.
ARCH_FLAG_SUBS = [
    ("-arch=ppu_10", GENCODE_SM80),
    ("-arch=ppu_15", GENCODE_SM80),
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


def apply_compile_chain(content: str) -> str:
    """Apply the general compile-chain rules to one file's content."""
    # Arch block first so the two-line pattern wins over per-flag subs.
    content = ARCH_BLOCK_RE.sub(
        lambda m: f"{m.group('indent')}{m.group('recv')}.append(\"{GENCODE_SM80}\")",
        content,
    )
    for old, new in ARCH_FLAG_SUBS + FLAG_SUBS:
        if old in content:
            content = content.replace(old, new)

    content = DROP_DEFINE_ITEM_RE.sub("", content)
    content = DROP_DEFINE_TEXT_RE.sub("", content)
    content = DROP_XHG_LIST_RE.sub("", content)
    content = DROP_XHG_TEXT_RE.sub("", content)

    for old, new in COMPILER_IDENTITY_SUBS:
        if old in content:
            content = content.replace(old, new)

    return content
