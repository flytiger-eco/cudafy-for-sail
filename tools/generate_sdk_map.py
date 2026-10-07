#!/usr/bin/env python3
"""Generate the PPU -> CUDA naming map (sdk_map.py).

Source of truth: the two header trees of the PPU SDK installation,

  - PPU side:   <PPU_SDK>/include (+ <PPU_SDK>/targets/x86_64-linux/include)
  - CUDA side:  <PPU_SDK>/CUDA_SDK/include

Every PPU-prefixed identifier found in the PPU headers is mapped through an
ordered list of prefix substitutions (hggc -> cuda, acblas -> cublas, ...) and
verified against the identifiers that actually exist on the CUDA side. Only
verified pairs are emitted; the rest is reported as unverified candidates.

Pairs that the conversion needs but the SDK trees cannot derive are carried
by the frozen curated constants below (HISTORICAL_TOKENS, COMPILER_MACROS,
DIAGNOSTIC_TEXT, SAILIFY_TOKENS, CURATED_PREFIX_OPEN, ...). The residual
cuda_compat scripts of the supported repositories own any further
repo-specific, non-SDK naming rules; this generator never reads them.

Emitted categories in sdk_map.py:

  SDK_* maps     -- pairs whose mapping is confirmed by the SDK pair
                    (directories, header files, functions, types, macros)
  NON_SDK_* maps -- pairs required by the conversion but NOT derivable from
                    the SDK trees: compiler-predefined macros, build artifact
                    names, diagnostic text, repo-local identifiers

Usage (inside the build environment that provides the PPU SDK):

  python3 tools/generate_sdk_map.py
"""

import os
import re
import sys
from pathlib import Path

PPU_INCLUDE_DIRS = [
    "/usr/local/PPU_SDK/include",
    "/usr/local/PPU_SDK/targets/x86_64-linux/include",
]
CUDA_INCLUDE_DIR = "/usr/local/PPU_SDK/CUDA_SDK/include"

OUT_DIR = Path(__file__).resolve().parent.parent
MAP_OUT = OUT_DIR / "sdk_map.py"
REPORT_OUT = OUT_DIR / "unmapped_sdk_tokens.txt"

HEADER_SUFFIXES = (".h", ".hpp")

# ---------------------------------------------------------------------------
# Ordered prefix substitutions applied to each extracted PPU token.
# ---------------------------------------------------------------------------
PREFIX_SUBS = [
    # cuBLAS family
    ("ACBLAS", "CUBLAS"),
    ("Acblas", "Cublas"),
    ("acBLAS", "cuBLAS"),
    ("acblas", "cublas"),
    # cuRAND family
    ("ACRAND", "CURAND"),
    ("acrand", "curand"),
    # cuDNN family
    ("ACDNN", "CUDNN"),
    ("acdnn", "cudnn"),
    # cuFFT family
    ("ACFFT", "CUFFT"),
    ("acfft", "cufft"),
    # cuSOLVER / cuSPARSE families
    ("ACSOLVER", "CUSOLVER"),
    ("acsolver", "cusolver"),
    ("ACSPARSE", "CUSPARSE"),
    ("acsparse", "cusparse"),
    # cuComplex family (types + make_/is_ helpers + acC* functions)
    ("make_ac", "make_cu"),
    ("is_ac", "is_cu"),
    ("acDoubleComplex", "cuDoubleComplex"),
    ("acFloatComplex", "cuFloatComplex"),
    ("acC", "cuC"),
    # NVRTC family (before generic HGGC/HG)
    ("HGRTC", "NVRTC"),
    ("hgrtc", "nvrtc"),
    # NVTX family
    ("HGTX", "NVTX"),
    ("hgtx", "nvtx"),
    ("hgToolsExt", "nvToolsExt"),
    # CUPTI family
    ("HGPTI", "CUPTI"),
    ("hgpti", "cupti"),
    # NPP / NvJPEG / NvML families
    ("HGPP", "NPP"),
    ("hgpp", "npp"),
    ("HGJPEG", "NVJPEG"),
    ("hgjpeg", "nvjpeg"),
    ("HGML", "NVML"),
    ("hgml", "nvml"),
    # CUDA runtime (hggcrt) family -- before generic hggc
    ("HGGCRT", "CUDART"),
    ("hggcrt", "cudart"),
    # CUDA core family
    ("HGGC", "CUDA"),
    ("hggc", "cuda"),
    # Device builtin types
    ("__ppu", "__nv"),
    ("__PPU", "__NV"),
    ("__hg_fp8", "__nv_fp8"),
    ("ppu_bfloat16", "nv_bfloat16"),
    # fatbin artifact naming
    ("HGBIN", "CUBIN"),
    ("hgbin", "cubin"),
    # CUDA driver family -- always last
    ("HG", "CU"),
    ("hg", "cu"),
]

# Extraction regexes: which PPU-side identifiers are mapping candidates.
TOKEN_REGEXES = [
    r"\b[Aa][Cc][Bb][Ll][Aa][Ss]\w*",
    r"\b[Aa][Cc][Rr][Aa][Nn][Dd]\w*",
    r"\b[Aa][Cc][Dd][Nn][Nn]\w*",
    r"\b[Aa][Cc][Ff][Ff][Tt]\w*",
    r"\b[Aa][Cc][Ss][Oo][Ll][Vv][Ee][Rr]\w*",
    r"\b[Aa][Cc][Ss][Pp][Aa][Rr][Ss][Ee]\w*",
    r"\bmake_ac[A-Z]\w*",
    r"\bis_ac[A-Z]\w*",
    r"\bacC[A-Za-z_]\w*",
    r"\bacDoubleComplex\w*",
    r"\bacFloatComplex\w*",
    r"\b[Hh][Gg][Rr][Tt][Cc]\w*",
    r"\b[Hh][Gg][Tt][Xx]\w*",
    r"\b[Hh][Gg][Pp][Tt][Ii]\w*",
    r"\b[Hh][Gg][Pp][Pp]\w*",
    r"\b[Hh][Gg][Jj][Pp][Ee][Gg]\w*",
    r"\b[Hh][Gg][Mm][Ll]\w*",
    r"\b[Hh][Gg][Gg][Cc][Rr][Tt]\w*",
    r"\b[Hh][Gg][Gg][Cc]\w*",
    r"\b[Hh][Gg][A-Za-z_]\w*",
    r"\b__ppu\w*",
    r"\b__PPU\w*",
    r"\b__hg_fp8\w*",
    r"\bppu_bfloat16\w*",
]

# Compiler-predefined macros: defined by the PPU frontend, consumed by user
# code, but never declared inside any SDK header.
COMPILER_MACROS = [
    ("__HGGCCC_RTC__", "__CUDACC_RTC__"),
    # SAIL_SDK version does not align with cudacc version
    # ("__HGGCCC_VER_MAJOR__", "__CUDACC_VER_MAJOR__"),
    # ("__HGGCCC_VER_MINOR__", "__CUDACC_VER_MINOR__"),
    ("__HGGCCC_VERSION__", "__CUDACC_VERSION__"),
    ("__HGGCCC__", "__CUDACC__"),
    ("__HGGC_NO_HALF_OPERATORS__", "__CUDA_NO_HALF_OPERATORS__"),
    ("__HGGC_NO_HALF_CONVERSIONS__", "__CUDA_NO_HALF_CONVERSIONS__"),
    ("__HGGC_NO_HALF2_OPERATORS__", "__CUDA_NO_HALF2_OPERATORS__"),
    ("__HGGC_NO_BFLOAT16_CONVERSIONS__", "__CUDA_NO_BFLOAT16_CONVERSIONS__"),
]

# Diagnostic phrases and build terminology renamed together with the API
# they describe. "CUDA-free" is the PPU-original wording; once the compiler
# is renamed to nvcc the wording must become "CUDA-compat" as well.
DIAGNOSTIC_TEXT = [
    ("HGGC error", "CUDA error"),
    ("HGGC driver API", "CUDA driver API"),
    ("HG driver error", "CUDA driver error"),
    ("CUDA-free", "CUDA-compat"),
    ("CUDA-Free", "CUDA-Compat"),
]

# SAILify platform naming: the COMPATIBLE_ARCH / __COMPATIBLECC_* family is
# introduced by the SAIL porting layer (not by the PPU SDK) and is shared by
# the SAIL-ified repositories. COMPATIBLE_ARCH must map to __CUDA_ARCH__: the
# compat build exposes 800/890 there, while leaving COMPATIBLE_ARCH undefined
# under nvcc would turn guarded autogen kernels into empty functions.
SAILIFY_TOKENS = [
    ("COMPATIBLE_ARCH", "__CUDA_ARCH__"),
    ("__COMPATIBLECC_VER_MAJOR__", "__CUDACC_VER_MAJOR__"),
]

# Directory-name / include-path-fragment mappings.
SDK_DIRECTORY_SEED = [
    ("hgtx3", "nvtx3"),
    ("hggc/std/", "cuda/std/"),
]

SDK_DIRECTORY_SEED_MAP = dict(SDK_DIRECTORY_SEED)
DIAGNOSTIC_TEXT_MAP = dict(DIAGNOSTIC_TEXT)
SAILIFY_TOKEN_MAP = dict(SAILIFY_TOKENS)

# Prefix-open pairs: applied as plain prefix substitutions (camelCase
# continuation allowed, no word boundary needed). Mirrors the deliberately
# prefix-only entries of the historical per-repo scripts.
CURATED_PREFIX_OPEN = [
    ("acrand", "curand"),
    ("HGBIN", "CUBIN"),
    ("hgbin", "cubin"),
    ("acBLAS", "cuBLAS"),
    ("ACBLAS", "CUBLAS"),
    ("Acblas", "Cublas"),
    ("acblas", "cublas"),
    ("acblasSgemm", "cublasSgemm"),
    ("acblasDgemm", "cublasDgemm"),
    ("acblasHgemm", "cublasHgemm"),
    ("acblasCgemm", "cublasCgemm"),
    ("acblasZgemm", "cublasZgemm"),
    ("__HGGC_STD_", "__NV_STD_"),
]

# Tokens that must never be mapped:
#   __HGGC_ARCH__ -- keeps the PPU arch numbering (100/150) predefined by the
#                    compat compiler; renaming would change guarded values.
#   __HGGC__      -- the compat compiler DEFINES __HGGC__ but does NOT define
#                    __CUDA__, so the rename would silently disable every
#                    `#if defined(__HGGC__)` region in device code.
NEVER_MAP = {
    "__HGGC_ARCH__",
    "__hggc_arch__",
    "__HGGC__",
    "__hggc__",
}

# Keys too broad to keep: '__ppu_' as a prefix-open key would rewrite PPU-only
# device intrinsics (__ppu_sgmdf, __ppu_barrier_sync_nocnt, ...) that the
# CUDA_SDK still ships under their original names via include/hgrt/.
BLOCKED_KEYS = {
    "__ppu_",
    "__PPU_",
}

# hgrand_* is a driver-side RNG namespace that does not correspond to curand
# one-to-one; none of the supported repositories use it.
EXCLUDED_TOKEN_PREFIXES = ("hgrand", "HGRAND")

# Historical token pairs, frozen once from the pre-general per-repo scripts
# (their generic token tables). These pairs are validated conversion
# behaviour but are NOT derivable from the SDK trees (repo-local identifiers,
# tokens absent from this SDK installation's headers, ...), so they stay here
# as plain data -- the generator does NOT parse any cuda_compat script.
# '__HGGC__' is deliberately absent (see NEVER_MAP).
HISTORICAL_TOKENS = [
    ("ACBLAS_ERROR", "CUBLAS_ERROR"),
    ("acBLAS", "cuBLAS"),
    ("BF16_HGRTC", "BF16_NVRTC"),
    ("DG_JIT_USE_HGRTC", "DG_JIT_USE_NVRTC"),
    ("FP8_HGRTC", "FP8_NVRTC"),
    ("HGGC_R_", "CUDA_R_"),
    ("INT8_HGRTC", "INT8_NVRTC"),
    ("__HGGC_STD_MAX", "__NV_STD_MAX"),
    ("__HGGC_STD_MIN", "__NV_STD_MIN"),
    ("__HGGC_STD_XYZ", "__NV_STD_XYZ"),
    ("get_hgcc_compiler", "get_nvcc_compiler"),
    ("hg_kernel", "cu_kernel"),
    ("hgbin_path", "cubin_path"),
    ("hggcErrorHggcrtUnloading", "cudaErrorCudartUnloading"),
    ("hggcEventElapsed", "cudaEventElapsed"),
    ("hggcGetDeviceProp", "cudaGetDeviceProp"),
    ("hggcTypedefs", "cudaTypedefs"),
    ("hggc_type_", "cuda_type_"),
    ("hggc_vector_types.h", "vector_types.h"),
    ("hgobjdump_path", "cuobjdump_path"),
    ("kernel.hgbin", "kernel.cubin"),
    ("lazy_hgFuncSetAttribute", "lazy_cuFuncSetAttribute"),
    ("lazy_hgGetErrorName", "lazy_cuGetErrorName"),
    ("lazy_hgGetErrorString", "lazy_cuGetErrorString"),
    ("lazy_hgLaunchKernelEx", "lazy_cuLaunchKernelEx"),
    ("lazy_hgModuleGetFunction", "lazy_cuModuleGetFunction"),
    ("lazy_hgModuleLoad", "lazy_cuModuleLoad"),
    ("lazy_hgModuleUnload", "lazy_cuModuleUnload"),
    ("lazy_hgTensorMapEncodeTiled", "lazy_cuTensorMapEncodeTiled"),
    ("libhggc.so", "libcuda.so.1"),
    ("to_ppu_bfloat16", "to_nv_bfloat16"),
    ("use_hgtx_", "use_nvtx_"),
    ("is_acComplex", "is_cuComplex"),
]

def strip_comments(text: str) -> str:
    """Remove C/C++ comments; comments carry no symbols."""
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", " ", text)
    return text


# Implementation-copy headers whose SYMBOLS are still the user-facing API:
# hgrtc.h declares the whole NVRTC-compatible API (hgrtcCreateProgram, ...)
# and no other PPU header does, so its symbols must stay extractable even
# though the file is a byte-copy of the compiler-internal one.
IMPL_COPY_SYMBOL_EXCEPTIONS = {"hgrtc.h"}


def compiler_impl_copies(files):
    """Headers that are byte-identical copies of the compiler-implementation
    headers under hgrt/details/system/ (e.g. hgComplex.h, hgrtc.h).

    The SDK duplicates some of those internal headers at the include root.
    Their SYMBOLS (hgCrealf, hgFloatComplex, ...) are compiler internals and
    must not seed the map; their FILE NAMES stay valid include-mapping
    candidates (e.g. <hgrtc.h> -> <nvrtc.h>).
    """
    import hashlib
    impl = {}
    for fp in files:
        if os.path.join("hgrt", "details", "system") in str(fp):
            try:
                impl.setdefault(fp.name, set()).add(
                    hashlib.md5(fp.read_bytes()).hexdigest())
            except OSError:
                pass
    copies = set()
    for fp in files:
        if os.path.join("hgrt", "details", "system") in str(fp):
            continue
        try:
            digest = hashlib.md5(fp.read_bytes()).hexdigest()
        except OSError:
            continue
        if digest in impl.get(fp.name, set()) \
                and fp.name not in IMPL_COPY_SYMBOL_EXCEPTIONS:
            copies.add(fp)
    return copies


def collect_headers(root: str):
    files = []
    for base, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(dirnames)
        for fname in sorted(filenames):
            if fname.endswith(HEADER_SUFFIXES):
                files.append(Path(base) / fname)
    return files


def identifier_set(files):
    tokens = set()
    for fp in files:
        try:
            text = strip_comments(fp.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
        tokens.update(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text))
    return tokens


def substitute(token: str):
    for old, new in PREFIX_SUBS:
        if old in token:
            token = token.replace(old, new)
    return token


def header_candidates(name: str):
    """Candidate CUDA header names for one PPU header file name.

    Prefix substitution first (hggc_bf16.h -> cuda_bf16.h), then prefix
    dropping (hggcrt_driver_types.h -> driver_types.h, hggc_mma.h -> mma.h).
    """
    cands = [substitute(name)]
    for prefix in ("hggcrt_", "hggc_"):
        if name.startswith(prefix):
            cands.append(name[len(prefix):])
    return cands




# ---------------------------------------------------------------------------
# Category classification
# ---------------------------------------------------------------------------

def classify_sdk_token(token: str) -> str:
    """Bucket for a pair whose mapping is confirmed by the SDK pair."""
    if token in SDK_DIRECTORY_SEED_MAP or "/" in token:
        return "SDK_DIRECTORY_MAP"
    if token.endswith(HEADER_SUFFIXES):
        return "SDK_HEADER_MAP"
    if token.startswith(("__ppu", "__PPU", "__hg_fp8")) or token.startswith("ppu_bfloat16"):
        return "SDK_TYPE_MAP"       # device builtin types
    if token.startswith("__") or re.fullmatch(r"[A-Z][A-Z0-9_]*", token):
        return "SDK_MACRO_MAP"      # macros and enum constants
    if token.endswith("_t"):
        return "SDK_TYPE_MAP"
    if re.match(r"^HG[A-Z][a-z]", token):
        return "SDK_TYPE_MAP"       # driver handle types (HGresult, ...)
    return "SDK_FUNCTION_MAP"


def classify_non_sdk(token: str) -> str:
    """Bucket for a pair required by the conversion but absent from the SDKs."""
    if " " in token:
        return "NON_SDK_TEXT_MAP"          # diagnostic phrases
    if token.startswith("__"):
        return "NON_SDK_COMPILER_MACROS"    # predefined-macro family
    if token.endswith((".so", ".hgbin", ".cubin")) or "hgbin" in token.lower() \
            or token in ("hggcrt1", "hgrtc"):
        return "NON_SDK_ARTIFACT_MAP"       # binaries, sonames, link libs
    return "NON_SDK_TOKEN_MAP"              # repo-local identifiers


def main() -> int:
    required_dirs = [*PPU_INCLUDE_DIRS, CUDA_INCLUDE_DIR]
    missing_dirs = [path for path in required_dirs if not os.path.isdir(path)]
    if missing_dirs:
        print("ERROR: required SDK include directories are missing:", file=sys.stderr)
        for path in missing_dirs:
            print(f"  {path}", file=sys.stderr)
        return 1

    ppu_raw = []
    for root in PPU_INCLUDE_DIRS:
        ppu_raw.extend(collect_headers(root))
    impl_copies = compiler_impl_copies(ppu_raw)
    ppu_all = []
    seen = set()
    for fp in ppu_raw:
        if fp.name not in seen:
            seen.add(fp.name)
            ppu_all.append(fp)
    # Symbols are extracted from the user-facing headers only; file names of
    # every header (including the implementation copies) stay candidates.
    ppu_files = [fp for fp in ppu_all if fp not in impl_copies]
    cuda_files = collect_headers(CUDA_INCLUDE_DIR)

    print(f"PPU headers:  {len(ppu_all)} ({len(ppu_all) - len(ppu_files)} "
          f"compiler-implementation copies excluded from symbol extraction)")
    print(f"CUDA headers: {len(cuda_files)}")

    cuda_tokens = identifier_set(cuda_files)
    # Full PPU-side identifier set (implementation copies included), used to
    # place curated compiler macros in the SDK category when both sides are
    # actually findable in the SDK trees.
    ppu_all_ids = identifier_set(ppu_all)
    cuda_file_names = {fp.name for fp in cuda_files}

    # -- extract candidate tokens from the PPU side ------------------------
    token_regex = re.compile("|".join(TOKEN_REGEXES))
    candidates = {}
    for fp in ppu_files:
        try:
            text = strip_comments(fp.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
        for match in token_regex.findall(text):
            candidates.setdefault(match, set()).add(fp.name)
    for fp in ppu_all:
        candidates.setdefault(fp.name, {"(filename)"})

    print(f"Candidate tokens: {len(candidates)}")

    verified = {}
    unverified = {}
    for token, sources in sorted(candidates.items()):
        if token in NEVER_MAP or token.startswith(EXCLUDED_TOKEN_PREFIXES):
            continue
        if token.endswith(HEADER_SUFFIXES):
            picked = None
            for cand in header_candidates(token):
                if cand != token and cand in cuda_file_names:
                    picked = cand
                    break
            if picked is not None:
                verified[token] = picked
            else:
                unverified[token] = (substitute(token), sorted(sources))
            continue
        sub = substitute(token)
        if sub == token:
            continue
        if sub in cuda_tokens:
            verified[token] = sub
        else:
            unverified[token] = (sub, sorted(sources))

    for blocked in BLOCKED_KEYS:
        verified.pop(blocked, None)

    # -- merge the frozen historical pairs -----------------------------------
    # A merged pair joins the SDK_* maps when both sides exist in the SDK
    # trees; otherwise it becomes a NON_SDK_* entry.
    for token, sub in sorted(HISTORICAL_TOKENS):
        if token in NEVER_MAP or token.startswith(EXCLUDED_TOKEN_PREFIXES):
            print(f"WARNING: blocked historical pair ignored: {token} -> {sub}")
            continue
        if token in verified:
            if verified[token] != sub:
                print(f"WARNING: SDK pair says {token} -> {verified[token]}, "
                      f"historical tokens say {token} -> {sub}; keeping the latter")
                verified[token] = sub
            continue
        verified[token] = sub
        # category-b classification is decided below by the SDK-membership
        # test; historical entries are applied regardless and must NOT show
        # up in the report's unverified-candidates list.

    # -- curated additions --------------------------------------------------
    curated = list(DIAGNOSTIC_TEXT) + list(SDK_DIRECTORY_SEED) + list(SAILIFY_TOKENS)
    for token, sub in curated:
        verified[token] = sub
    # Compiler-predefined macros join the SDK category when both sides are
    # findable in the SDK trees; the rest stays NON_SDK by design.
    sdk_macros = {t: s for t, s in COMPILER_MACROS
                  if t in ppu_all_ids and s in cuda_tokens}
    non_sdk_macros = {t: s for t, s in COMPILER_MACROS if t not in sdk_macros}
    for token, sub in sdk_macros.items():
        verified[token] = sub
    for token, sub in non_sdk_macros.items():
        verified[token] = sub

    # Split every pair into the SDK-confirmed (a) and non-SDK (b) halves. A
    # pair is SDK-confirmed when the token occurs in the PPU headers and the
    # substituted name on the CUDA side (identifier or header file name).
    sdk_maps = {name: {} for name in
                ("SDK_DIRECTORY_MAP", "SDK_HEADER_MAP", "SDK_FUNCTION_MAP",
                 "SDK_TYPE_MAP", "SDK_MACRO_MAP")}
    non_sdk_maps = {name: {} for name in
                    ("NON_SDK_COMPILER_MACROS", "NON_SDK_ARTIFACT_MAP",
                     "NON_SDK_TEXT_MAP", "NON_SDK_TOKEN_MAP",
                     "NON_SDK_SAILIFY_MAP")}

    ppu_all_tokens = set(candidates)
    for token, sub in verified.items():
        if token in ppu_all_tokens and (sub in cuda_tokens or sub in cuda_file_names):
            sdk_maps[classify_sdk_token(token)][token] = sub
        elif token in SDK_DIRECTORY_SEED_MAP:
            sdk_maps["SDK_DIRECTORY_MAP"][token] = sub
        elif token in sdk_macros:
            sdk_maps["SDK_MACRO_MAP"][token] = sub
        elif token in non_sdk_macros:
            non_sdk_maps["NON_SDK_COMPILER_MACROS"][token] = sub
        elif token in SAILIFY_TOKEN_MAP:
            non_sdk_maps["NON_SDK_SAILIFY_MAP"][token] = sub
        elif token in DIAGNOSTIC_TEXT_MAP:
            non_sdk_maps["NON_SDK_TEXT_MAP"][token] = sub
        else:
            non_sdk_maps[classify_non_sdk(token)][token] = sub

    # Prefix-open: curated entries plus verified bare prefixes that must
    # match camelCase continuations.
    sdk_prefix_open = {}
    non_sdk_prefix_open = {}
    for token, sub in CURATED_PREFIX_OPEN:
        if token in ppu_all_tokens and (sub in cuda_tokens or sub in cuda_file_names):
            sdk_prefix_open[token] = sub
        else:
            non_sdk_prefix_open[token] = sub

    # -- emit sdk_map.py -----------------------------------------------------
    lines = []
    lines.append('"""Generated PPU -> CUDA naming map (DO NOT EDIT BY HAND).')
    lines.append("")
    lines.append("Produced by tools/generate_sdk_map.py from the PPU SDK")
    lines.append("header trees (include/ and CUDA_SDK/include).")
    lines.append("")
    lines.append("Map categories, in file order:")
    lines.append("  1-5    SDK_*     -- pairs confirmed by the SDK trees")
    lines.append("                   (directories, header files, functions, types,")
    lines.append("                   macros / enum constants)")
    lines.append("  6-10   NON_SDK_* -- pairs required by the conversion but not")
    lines.append("                   derivable from the SDK trees: compiler-predefined")
    lines.append("                   macros, build artifact names, diagnostic text,")
    lines.append("                   repo-local identifiers, SAILify platform naming")
    lines.append("  11-12  prefix-open maps (camelCase continuation allowed)")
    lines.append("")
    lines.append("The engine merges every map into one exact-token lookup plus")
    lines.append("the two prefix-open maps; declaration order carries no meaning.")
    lines.append("Regenerate with: python3 tools/generate_sdk_map.py")
    lines.append('"""')
    lines.append("")

    def emit_dict(name, mapping, comment):
        lines.append(f"# {comment}")
        lines.append(f"{name} = {{")
        for token in sorted(mapping, key=len, reverse=True):
            lines.append(f"    {token!r}: {mapping[token]!r},")
        lines.append("}")
        lines.append("")

    emit_dict("SDK_DIRECTORY_MAP", sdk_maps["SDK_DIRECTORY_MAP"],
              "Category 1 -- directory names and include-path fragments.")
    emit_dict("SDK_HEADER_MAP", sdk_maps["SDK_HEADER_MAP"],
              "Category 2 -- header file names (include rewrites).")
    emit_dict("SDK_FUNCTION_MAP", sdk_maps["SDK_FUNCTION_MAP"],
              "Category 3 -- API functions and symbols.")
    emit_dict("SDK_TYPE_MAP", sdk_maps["SDK_TYPE_MAP"],
              "Category 4 -- types, handles and device builtins.")
    emit_dict("SDK_MACRO_MAP", sdk_maps["SDK_MACRO_MAP"],
              "Category 5 -- macros and enum constants.")
    emit_dict("NON_SDK_COMPILER_MACROS", non_sdk_maps["NON_SDK_COMPILER_MACROS"],
              "Category 6 -- compiler-predefined macros (not in SDK headers).")
    emit_dict("NON_SDK_ARTIFACT_MAP", non_sdk_maps["NON_SDK_ARTIFACT_MAP"],
              "Category 7 -- build artifacts, sonames and link libraries.")
    emit_dict("NON_SDK_TEXT_MAP", non_sdk_maps["NON_SDK_TEXT_MAP"],
              "Category 8 -- diagnostic text.")
    emit_dict("NON_SDK_TOKEN_MAP", non_sdk_maps["NON_SDK_TOKEN_MAP"],
              "Category 9 -- repo-local identifiers (e.g. DG_JIT_USE_HGRTC).")
    emit_dict("NON_SDK_SAILIFY_MAP", non_sdk_maps["NON_SDK_SAILIFY_MAP"],
              "Category 10 -- SAILify platform naming (COMPATIBLE_ARCH family).")
    emit_dict("SDK_PREFIX_OPEN", sdk_prefix_open,
              "Category 11 -- prefix-open, SDK-confirmed (camelCase continuation allowed).")
    emit_dict("NON_SDK_PREFIX_OPEN", non_sdk_prefix_open,
              "Category 12 -- prefix-open, non-SDK (camelCase continuation allowed).")

    MAP_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    n_sdk = sum(len(m) for m in sdk_maps.values()) + len(sdk_prefix_open)
    n_total = (n_sdk + sum(len(m) for m in non_sdk_maps.values()) +
               len(non_sdk_prefix_open))
    print(f"Wrote {MAP_OUT} ({n_total} entries: {n_sdk} SDK-confirmed, "
          f"{n_total - n_sdk} non-SDK)")

    # -- emit review report --------------------------------------------------
    with open(REPORT_OUT, "w", encoding="utf-8") as rpt:
        rpt.write("SDK map generation report\n")
        rpt.write("========================\n\n")
        rpt.write(f"emitted entries      : {n_total} "
                  f"(SDK-confirmed {n_sdk}, non-SDK {n_total - n_sdk})\n")
        rpt.write(f"unverified candidates: {len(unverified)}\n\n")
        rpt.write("Unverified candidates: PPU-side tokens for which the\n")
        rpt.write("prefix substitution produced a name that does NOT exist on the\n")
        rpt.write("CUDA side. Kept here for future reference only -- they are NOT\n")
        rpt.write("part of the applied map.\n\n")
        for token, (sub, sources) in sorted(unverified.items()):
            rpt.write(f"  {token!r} -> {sub!r}   [{sources[0]}]\n")
    print(f"Wrote {REPORT_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
