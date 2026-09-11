#!/usr/bin/env python3
"""
cuda_compat.py — Convert ACTLIZE v0.5.0 to CUDA-compatible version


Usage:
    python3 cuda_compat_v0.5.0.py <include_dir>

Processing contents:
    1. Global text replacement (NAMING_REPLACEMENTS)
       - Runtime API functions (hggc* → cuda*)
       - Runtime types (hggcStream_t → cudaStream_t, etc.)
       - Driver / TMA symbols (HGtensorMap → CUtensorMap, etc.)
       - Device built-in types (__ppu_bfloat16 → __nv_bfloat16, etc.)
       - System header includes (<hggc_*.h> → <cuda_*.h>, etc.)
       - namespace (hggc:: → cuda::)
       - Compiler macros (HGGCCC → CUDACC, etc.)
       - Note: __HGGC_ARCH__ is NOT replaced; original PPU arch numbers (100/150) are preserved
"""

import sys
import os
import re

# ============================================================
# NAMING_REPLACEMENTS (PPU/HGGC → CUDA); order matters: longer patterns before shorter ones
# ============================================================
NAMING_REPLACEMENTS = [
    # === Category 1: Device built-in types (longer patterns first) ===
    (r'\b__ppu_bfloat16_raw\b', '__nv_bfloat16_raw'),
    (r'\b__ppu_bfloat162\b', '__nv_bfloat162'),
    (r'\bto_ppu_bfloat16\b', 'to_nv_bfloat16'),
    (r'\b__ppu_bfloat16\b', '__nv_bfloat16'),
    (r'\b__ppu_fp8_e4m3\b', '__nv_fp8_e4m3'),
    (r'\b__ppu_fp8_e5m2\b', '__nv_fp8_e5m2'),
    (r'\b__hg_fp8_e4m3\b', '__nv_fp8_e4m3'),
    (r'\b__hg_fp8_e5m2\b', '__nv_fp8_e5m2'),
    (r'\b__hg_fp8_storage_t\b', '__nv_fp8_storage_t'),

    # === Category 2: CUDA Driver / TMA symbols ===
    (r'\bHGtensorMapDataType\b', 'CUtensorMapDataType'),
    (r'\bHGtensorMapSwizzle\b', 'CUtensorMapSwizzle'),
    (r'\bHGtensorMapInterleave\b', 'CUtensorMapInterleave'),
    (r'\bHGtensorMapL2promotion\b', 'CUtensorMapL2promotion'),
    (r'\bHGtensorMapFloatOOBfill\b', 'CUtensorMapFloatOOBfill'),
    (r'\bhgTensorMapEncodeTiled\b', 'cuTensorMapEncodeTiled'),
    (r'\bhgMemsetD32Async\b', 'cuMemsetD32Async'),
    (r'\bhgMemsetD16Async\b', 'cuMemsetD16Async'),
    (r'\bhgMemsetD8Async\b', 'cuMemsetD8Async'),
    (r'\bhgGetErrorString\b', 'cuGetErrorString'),
    (r'\bHGtensorMap\b', 'CUtensorMap'),
    (r'\bHGresult\b', 'CUresult'),
    (r'\bHGdeviceptr\b', 'CUdeviceptr'),

    # === Category 3: CUDA Runtime API functions (longer patterns first) ===
    (r'\bhggcOccupancyMaxActiveBlocksPerMultiprocessorWithFlags\b', 'cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags'),
    (r'\bhggcOccupancyMaxActiveBlocksPerMultiprocessor\b', 'cudaOccupancyMaxActiveBlocksPerMultiprocessor'),
    (r'\bhggcFuncAttributePreferredSharedMemoryCarveout\b', 'cudaFuncAttributePreferredSharedMemoryCarveout'),
    (r'\bhggcFuncAttributeMaxDynamicSharedMemorySize\b', 'cudaFuncAttributeMaxDynamicSharedMemorySize'),
    (r'\bhggcLaunchAttributeProgrammaticStreamSerialization\b', 'cudaLaunchAttributeProgrammaticStreamSerialization'),
    (r'\bhggcGetDriverEntryPointByVersion\b', 'cudaGetDriverEntryPointByVersion'),
    (r'\bhggcDriverEntryPointQueryResult\b', 'cudaDriverEntryPointQueryResult'),
    (r'\bhggcOccupancyDisableCachingOverride\b', 'cudaOccupancyDisableCachingOverride'),
    (r'\bhggcDeviceGetAttribute\b', 'cudaDeviceGetAttribute'),
    (r'\bhggcDeviceSynchronize\b', 'cudaDeviceSynchronize'),
    (r'\bhggcGetErrorString\b', 'cudaGetErrorString'),
    (r'\bhggcGetDriverEntryPoint\b', 'cudaGetDriverEntryPoint'),
    (r'\bhggcDriverEntryPointSuccess\b', 'cudaDriverEntryPointSuccess'),
    (r'\bhggcMemcpyHostToDevice\b', 'cudaMemcpyHostToDevice'),
    (r'\bhggcGetLastError\b', 'cudaGetLastError'),
    (r'\bhggcPeekAtLastError\b', 'cudaPeekAtLastError'),
    (r'\bhggcFuncSetAttribute\b', 'cudaFuncSetAttribute'),
    (r'\bhggcFuncGetAttributes\b', 'cudaFuncGetAttributes'),
    (r'\bhggcFuncAttributes\b', 'cudaFuncAttributes'),
    (r'\bhggcDevAttrMultiProcessorCount\b', 'cudaDevAttrMultiProcessorCount'),
    (r'\bhggcDevAttrClockRate\b', 'cudaDevAttrClockRate'),
    (r'\bhggcMemsetAsync\b', 'cudaMemsetAsync'),
    (r'\bhggcLaunchKernelEx\b', 'cudaLaunchKernelEx'),
    (r'\bhggcLaunchKernel\b', 'cudaLaunchKernel'),
    (r'\bhggcLaunchAttribute\b', 'cudaLaunchAttribute'),
    (r'\bhggcLaunchConfig_t\b', 'cudaLaunchConfig_t'),
    (r'\bhggcMemcpyToSymbol\b', 'cudaMemcpyToSymbol'),
    (r'\bhggcGetDeviceCount\b', 'cudaGetDeviceCount'),
    (r'\bhggcGetErrorName\b', 'cudaGetErrorName'),
    (r'\bhggcEnableDefault\b', 'cudaEnableDefault'),
    (r'\bhggcGetDevice\b', 'cudaGetDevice'),
    (r'\bhggcSetDevice\b', 'cudaSetDevice'),
    (r'\bhggcMemcpy\b', 'cudaMemcpy'),
    (r'\bhggcMemset\b', 'cudaMemset'),
    (r'\bhggcMalloc\b', 'cudaMalloc'),
    (r'\bhggcTypedefs\b', 'cudaTypedefs'),
    (r'\bhggcDeviceProp\b', 'cudaDeviceProp'),
    (r'\bhggcGetDeviceProperties\b', 'cudaGetDeviceProperties'),
    (r'\bhggcMemcpyKind\b', 'cudaMemcpyKind'),
    (r'\bhggcFree\b', 'cudaFree'),
    (r'\bhggcMemcpyDeviceToHost\b', 'cudaMemcpyDeviceToHost'),
    (r'\bhggcMemcpyHostToHost\b', 'cudaMemcpyHostToHost'),
    (r'\bhggcMemcpyDeviceToDevice\b', 'cudaMemcpyDeviceToDevice'),
    (r'\bhggcMemcpyDefault\b', 'cudaMemcpyDefault'), 

    # === Category 4: CUDA Runtime types ===
    (r'\bhggcStreamNonBlocking\b', 'cudaStreamNonBlocking'),
    (r'\bhggcStreamDefault\b', 'cudaStreamDefault'),
    (r'\bhggcStream_t\b', 'cudaStream_t'),
    (r'\bhggcSuccess\b', 'cudaSuccess'),
    (r'\bhggcError_t\b', 'cudaError_t'),

    # === Category 5: System header includes (hggc → cuda; omit <> to handle both quote styles) ===
    (r'\bhggc_runtime_api\.h\b', 'cuda_runtime_api.h'),
    (r'\bhggc_runtime\.h\b', 'cuda_runtime.h'),
    (r'\bhggc_fp16\.h\b', 'cuda_fp16.h'),
    (r'\bhggc_fp8\.h\b', 'cuda_fp8.h'),
    (r'\bhggc_bf16\.h\b', 'cuda_bf16.h'),
    (r'\bacComplex\.h\b', 'cuComplex.h'),
    (r'\bhgComplex\.h\b', 'cuComplex.h'),
    (r'\bhggc_pipeline\.h\b', 'cuda_pipeline.h'),
    (r'\bhggc_mma\.h\b', 'mma.h'),

    # === ac* complex type → cu* mapping (long patterns first) ===
    (r'\bmake_acFloatComplex\b', 'make_cuFloatComplex'),
    (r'\bmake_acDoubleComplex\b', 'make_cuDoubleComplex'),
    (r'\bacCrealf\b', 'cuCrealf'),
    (r'\bacCimagf\b', 'cuCimagf'),
    (r'\bacCreal\b', 'cuCreal'),
    (r'\bacCimag\b', 'cuCimag'),
    (r'\bacFloatComplex\b', 'cuFloatComplex'),
    (r'\bacDoubleComplex\b', 'cuDoubleComplex'),
    (r'\bhggcTypedefs\.h\b', 'cudaTypedefs.h'),
    (r'\bhggc\.h\b', 'cuda.h'),
    (r'\bhggcrt_driver_types\.h\b', 'driver_types.h'),
    (r'\bhggc/std/cassert\b', 'cuda/std/cassert'),
    (r'\bhggc/std/cstdint\b', 'cuda/std/cstdint'),
    (r'\bhggc/std/cstddef\b', 'cuda/std/cstddef'),
    (r'\bhggc/std/utility\b', 'cuda/std/utility'),
    (r'\bhggc/std/type_traits\b', 'cuda/std/type_traits'),
    (r'\bhggc/std/tuple\b', 'cuda/std/tuple'),
    (r'\bhggc/std/limits\b', 'cuda/std/limits'),

    # === M4: CUTLASS internal macros + compiler macro mapping ===
    # Host adapter: type rename → alias injection, file rename → thin wrapper, param/local → no change

    # Driver macros + types
    (r'\bHGlaunchAttribute\b', 'CUlaunchAttribute'),
    (r'\bhgTensorMapEncodeIm2col\b', 'cuTensorMapEncodeIm2col'),

    # Barrier + Clang (internal macros, not converted)

    # Platform macros
    (r'\b__HGGC_STD_MAX\b', '__NV_STD_MAX'),
    (r'\b__HGGC_STD_MIN\b', '__NV_STD_MIN'),


    # CUDA API constants
    (r'\bHGGC_SUCCESS\b', 'CUDA_SUCCESS'),
    (r'\bHGGC_ERROR_UNKNOWN\b', 'CUDA_ERROR_UNKNOWN'),
    (r'\bhggcErrorUnknown\b', 'cudaErrorUnknown'),

    # Compiler/runtime version macros (HGGC → CUDA, longer patterns first)
    (r'\b__HGGCCC_RTC__\b', '__CUDACC_RTC__'),
    (r'\b__HGGCCC_VER_MAJOR__\b', '__CUDACC_VER_MAJOR__'),
    (r'\b__HGGCCC_VER_MINOR__\b', '__CUDACC_VER_MINOR__'),
    (r'\bHGGCRT_VERSION\b', 'CUDART_VERSION'),
    (r'\b__HGGCCC_VERSION__\b', '__CUDACC_VERSION__'),
    (r'\b__HGGCCC__\b', '__CUDACC__'),

    # Clang CUDA macro
    (r'\b__HGGC__\b', '__CUDA__'),
    # __HGGC_ARCH__ always matches __CUDA_ARCH__ in compatible compiler    

    # === namespace (hggc:: → cuda::) ===
    (r'\bhggc::', 'cuda::'),
]


def apply_naming_replacements(content):
    """Apply conversion rules (PPU/HGGC → CUDA) to file content."""
    for pattern, replacement in NAMING_REPLACEMENTS:
        content = re.sub(pattern, replacement, content)
    return content


def process_directory(include_dir):
    """Walk the include/ directory and apply conversion to all C/C++ files."""
    changed_files = []
    total_files = 0

    # actlize/include dir
    for root, dirs, files in os.walk(include_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            total_files += 1

            # Process only text files (C/C++ header/source)
            if not any(fname.endswith(ext) for ext in
                       ('.h', '.hpp', '.cuh', '.cu', '.c', '.cpp', '.inl', '.inc')):
                continue

            try:
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                    original = f.read()
            except Exception:
                continue

            modified = apply_naming_replacements(original)

            if modified != original:
                with open(fpath, 'w', encoding='utf-8') as f:
                    f.write(modified)
                rel = os.path.relpath(fpath, include_dir)
                changed_files.append(rel)

    # Also process ../tools/util/include relative to include_dir
    tools_util_include = os.path.normpath(os.path.join(include_dir, '..', 'tools', 'util', 'include'))
    if os.path.isdir(tools_util_include):
        for root, dirs, files in os.walk(tools_util_include):
            for fname in files:
                fpath = os.path.join(root, fname)
                total_files += 1

                # Process only text files (C/C++ header/source)
                if not any(fname.endswith(ext) for ext in
                           ('.h', '.hpp', '.cuh', '.cu', '.c', '.cpp', '.inl', '.inc')):
                    continue

                try:
                    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                        original = f.read()
                except Exception:
                    continue

                modified = apply_naming_replacements(original)

                if modified != original:
                    with open(fpath, 'w', encoding='utf-8') as f:
                        f.write(modified)
                    rel = os.path.relpath(fpath, tools_util_include)
                    changed_files.append(f'tools/util/include/{rel}')

    return changed_files, total_files


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 cuda_compat.py <include_dir>")
        sys.exit(1)

    include_dir = sys.argv[1]
    if not os.path.isdir(include_dir):
        print(f"Error: directory not found: {include_dir}")
        sys.exit(1)

    print(f"[cuda_compat] include dir: {include_dir}")
    print("[cuda_compat] naming conversion")

    changed, total = process_directory(include_dir)

    # Handle file renames (none — all handled by wrappers)
    renames = []
    for src_rel, dst_rel in renames:
        src = os.path.join(include_dir, src_rel)
        dst = os.path.join(include_dir, dst_rel)
        if os.path.exists(src) and not os.path.exists(dst):
            os.rename(src, dst)
            changed.append(f"{src_rel} -> {dst_rel} (rename)")

    print(f"[cuda_compat] files scanned: {total}")
    print(f"[cuda_compat] files changed: {len(changed)}")
    for f in sorted(changed):
        print(f"  {f}")


if __name__ == "__main__":
    main()
