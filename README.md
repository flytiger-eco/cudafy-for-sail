# cudafy-for-sail

A toolkit for converting PPU original HGGC-based repositories to
CUDA-compatible source trees.

## How it works

All supported repositories use the shared conversion engine:

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

2. **Repository coverage** -- all supported repositories are covered by
   the general rules, with no per-repository residual scripts:

   | Repository | Residual rules |
   |---|---|
   | `actlize` (0.5.0 / 0.8.0 / 1.0.0) | none -- fully covered by the general rules |
   | `flash-attention` (2.7.2 / 2.7.4 / 2.8.2) | none -- fully covered by the general rules |
   | `xformers` (0.0.27) | none -- the `COMPATIBLE_ARCH` / `__COMPATIBLECC_VER_*` SAILify platform naming lives in the general map (`NON_SDK_SAILIFY_MAP`) |
   | `flashmla` | none -- fully covered by the general rules |
   | `deepgemm` | none -- fully covered by the general rules |

   Repository-specific commands retain their historical CLI shape and run
   the general engine directly.

## Usage

### Nightly vLLM SAIL wheels

[vLLM SAIL nightly wheels](.github/workflows/nightly-vllm-sail.yaml) builds
`vllm-project/vllm-sail` directly, without cudafy conversion, every day at
**21:00 Asia/Shanghai (13:00 UTC)**. The schedule becomes active once the
workflow is merged into this repository's default branch.

The workflow uses the existing Flytiger cluster runner and SAIL SDK image.
It runs upstream's `scripts/build_wheels.sh` to build the latest SAIL `main`
and the exact vLLM commit pinned by that checkout. Both `ppu_10` and `ppu_15`
are compiled in one build. The upstream tooling builds vLLM's Rust components,
requires all four SAIL native extensions and verifies the wheel manifest.
This is a build and packaging check; it does not run device or inference tests.

Wheels and their manifest are retained as GitHub Actions artifacts for 14 days.
After a successful build, the existing `push-tmp-wheel.yml` template publishes
both wheels to a timestamped temporary directory; the final job prints its URL
in the run summary. Build logs are saved as a separate diagnostics artifact,
including on build failure when logs are available.

Repository or organization Actions secrets `RELEASE_USERNAME` and
`RELEASE_TOKEN` must be available for upload. `TEMPLATE_TOKEN` is optional
unless access to the shared template repository requires it. The cluster
runner, shared NAS mounts and build image access follow the existing workflows.

Use **Actions → vLLM SAIL nightly wheels → Run workflow** for a manual build.
`target_ref` accepts a SAIL branch, tag, commit SHA or `refs/pull/<number>/head`
(default `main`). `sail_image` can select a matching CUDA-compatible or
CUDA-free SAIL SDK/PyTorch image; an empty value uses the existing Python 3.12,
SDK 2.2.0, PyTorch 2.13.0 image. The same inputs are exposed through
`workflow_call`.

### Conversion commands

Use `general` as the recommended entry point for all repositories.
Existing repository-specific commands remain supported and use the same
general engine.

Run from the repository root:

```bash
python3 cudafy.py --help
```

### General mode (any tree)

```bash
python3 cudafy.py general <path> [--dry-run] [--verbose]
```

Converts any path with the general rules only (SDK map + compile chain).
The general engine is also expected to work with other PPU libraries.
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

Both commands use the general engine only, equivalent to
`python3 cudafy.py general <path>`.

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
