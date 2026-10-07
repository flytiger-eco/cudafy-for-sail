#!/usr/bin/env python3
"""Audit sdk_map.py against the two SDK header trees.

For every SDK-side category (1-5 plus the SDK prefix-open map 11), each
entry's key must be findable in the PPU SDK user-side headers and its value
in the CUDA-side headers. Any violation is reported as a problem list.

NON_SDK_* categories (6-10, 12) are exempt by definition: their pairs are not
derivable from the SDK trees -- they come from the frozen curated constants
of tools/generate_sdk_map.py (compiler macros, artifact names, diagnostic
text, repo-local identifiers, SAILify platform naming).

Run inside the environment that provides the PPU SDK.
"""

import importlib.util
import os
import re
from pathlib import Path

MAP = Path(__file__).resolve().parent.parent / "sdk_map.py"
SDK_MAPS = ("SDK_DIRECTORY_MAP", "SDK_HEADER_MAP", "SDK_FUNCTION_MAP",
            "SDK_TYPE_MAP", "SDK_MACRO_MAP", "SDK_PREFIX_OPEN")

PPU_DIRS = ("/usr/local/PPU_SDK/include",
            "/usr/local/PPU_SDK/targets/x86_64-linux/include")
CUDA_DIR = "/usr/local/PPU_SDK/CUDA_SDK/include"


def strip_comments(t):
    t = re.sub(r"/\*.*?\*/", " ", t, flags=re.DOTALL)
    t = re.sub(r"//[^\n]*", " ", t)
    return t


def collect(root):
    files = []
    for base, dirs, names in os.walk(root):
        for n in names:
            if n.endswith((".h", ".hpp")):
                files.append(Path(base) / n)
    return files


def tokenset(files):
    s = set()
    for f in files:
        try:
            s.update(re.findall(r"[A-Za-z_][A-Za-z0-9_]*",
                                strip_comments(f.read_text(encoding="utf-8",
                                                           errors="ignore"))))
        except OSError:
            pass
    return s


def main():
    spec = importlib.util.spec_from_file_location("m", MAP)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    ppu_files = []
    seen = set()
    for r in PPU_DIRS:
        for f in collect(r):
            if f.name not in seen:
                seen.add(f.name)
                ppu_files.append(f)
    cuda_files = collect(CUDA_DIR)

    ppu_ids = tokenset(ppu_files)
    cuda_ids = tokenset(cuda_files)
    ppu_names = {f.name for f in ppu_files}
    cuda_names = {f.name for f in cuda_files}

    def findable(token, ids, names, roots):
        """A token is findable when it is an identifier, a header name, or
        (for path fragments) an existing directory under any SDK root."""
        if token in ids or token in names:
            return True
        return any(os.path.isdir(os.path.join(root, token.rstrip("/")))
                   for root in roots)

    print(f"PPU user-side headers: {len(ppu_files)} (identifiers: {len(ppu_ids)})")
    print(f"CUDA headers:          {len(cuda_files)} (identifiers: {len(cuda_ids)})")

    problems = []
    n = 0
    for cat in SDK_MAPS:
        for k, v in getattr(m, cat).items():
            n += 1
            if not findable(k, ppu_ids, ppu_names, PPU_DIRS):
                problems.append((cat, "KEY not findable in PPU SDK", k, v))
            elif not findable(v, cuda_ids, cuda_names, (CUDA_DIR,)):
                problems.append((cat, "VALUE not findable in CUDA SDK", k, v))
    print(f"SDK-side entries audited: {n}, violations: {len(problems)}")
    for p in problems:
        print("  VIOLATION:", p)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
