# cudafy-for-sail

A toolkit for converting PPU original HGGC-based repositories to
CUDA-compatible source trees.

## How it works

Every conversion is the sum of two layers:

1. **The general rules** (repository root)
   - `sdk_map.py` -- the full PPU -> CUDA naming map, generated from the
     PPU SDK header trees by `tools/generate_sdk_map.py`. Categories 1-5 are
     pairs confirmed against the SDK trees (directories, header files,
     functions, types, macros); categories 6-10 are pairs required by the
     conversion but not derivable from them (compiler-predefined macros,
     build artifact names, diagnostic text, repo-local identifiers such as
     `DG_JIT_USE_HGRTC`, SAILify platform naming); categories 11-12 are
     prefix-open maps.
   - `compile_chain.py` -- the shared compile-chain translation: `hgcc` ->
     `nvcc`, `-arch=ppu_10` -> `-gencode=arch=compute_80a,code=sm_80a`,
     `-arch=ppu_15` -> `-gencode=arch=compute_89,code=sm_89`
     (both flags are preserved when requested), `hgbin` -> `cubin`, removal of
     PPU-only `-D` defines.
   - `general.py` (the conversion engine) -- applies the above, renames
     files and directories whose names the map converts (so rewritten
     includes keep resolving), and preserves `#pragma hggc ...` lines and PPU-toolchain option values
     (e.g. `-no-hggc-embed-bc`) verbatim: the CUDA-compatible nvcc still
     honors them under their original spelling.

2. **Per-repo residual scripts** (`<repo>/cuda_compat*.py`) -- the rules the
   general engine cannot express. Each script runs the general engine first
   and then applies only its residual rules, so the residual file shows
   exactly what is still specific to that repository:

   | Repository | Residual rules |
   |---|---|
   | `actlize` (0.5.0 / 0.8.0 / 1.0.0) | none -- fully covered by the general rules |
   | `flash-attention` (2.7.2 / 2.7.4 / 2.8.2) | none -- fully covered by the general rules |
   | `xformers` (0.0.27) | none -- the `COMPATIBLE_ARCH` / `__COMPATIBLECC_VER_*` SAILify platform naming lives in the general map (`NON_SDK_SAILIFY_MAP`) |
   | `flashmla` | drop the forward-declared `cudaStream_t` typedef; link `libcuda`; enable the ptxas resource report |
   | `deepgemm` | JIT compiler rewrites (nvcc discovery, version query, flag-set translation, RTC include paths); `python_api.cpp` -> `.cu`; `#if defined(__HGGC__)` guard removal; setup.py / `__init__.py` SDK-root plumbing; CMakeLists.txt restoration |

   Repositories with no residual rules have no directory or script left; the
   goal is to shrink the residual set over time until every repository is
   covered by the general rules alone.

## Usage

Run from the repository root:

```bash
python3 cudafy.py --help
```

### General mode (any tree)

```bash
python3 cudafy.py general <path> [--dry-run] [--verbose]
```

Converts any path with the general rules only (SDK map + compile chain).
Bundled actlize copies under the path are converted together with the tree,
so no separate actlize step is needed.

Any command whose repository name is not recognized falls back to this mode:
`python3 cudafy.py <unknown-repo> <path>` is executed as
`python3 cudafy.py general <path>`.

### ACTLIZE

```bash
python3 cudafy.py actlize [--version=1.0.0] /path/to/actlize/include
```

The include directory's `tools/util/include` sibling is converted as well.
`--version` is informational only: the general rules are version-independent.

### Flash-Attention

```bash
python3 cudafy.py flash-attention --version=2.8.2 /path/to/flash-attention
```

Supported versions: `2.7.2`, `2.7.4`, `2.8.2` (informational only). FA2 and
FA3 (`hopper/`) are converted in one run.

### xFormers

```bash
python3 cudafy.py xformers --version=0.0.27 /path/to/xformers
```

`--version` is informational only: the general rules are version-independent.

### DeepGEMM / FlashMLA

```bash
python3 cudafy.py deepgemm /path/to/DeepGEMM
python3 cudafy.py flashmla /path/to/FlashMLA
```

These run the general rules plus the per-repo residual scripts listed above.

### Regenerating the SDK map

After a PPU SDK update, regenerate `sdk_map.py` inside an environment that
provides the SDK:

```bash
python3 tools/generate_sdk_map.py
```

`unmapped_sdk_tokens.txt` lists the tokens for which no CUDA-side
counterpart was found; they are kept for future reference only and are not
part of the applied map.

Check the generated map against the SDK trees (SDK categories only: each
key must be findable in the PPU SDK headers, each value in the CUDA SDK
headers; NON_SDK categories come from the generator's frozen curated
constants and are exempt):

```bash
python3 tools/audit_sdk_map.py
```
