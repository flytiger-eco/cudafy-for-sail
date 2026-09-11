#!/usr/bin/env python3
"""
cuda_compat.py — Convert PPU ACTLIZE v0.8.0 to CUDA-compatible version

Usage:
    python3 cuda_compat_v0.8.0.py <include_dir>

Processing contents:
    1. Global text replacement (NAMING_REPLACEMENTS)
       - Runtime API functions (hggc* → cuda*)
       - Runtime types (hggcStream_t → cudaStream_t, etc.)
       - Driver symbols (HGresult → CUresult, hgMemsetD32Async → cuMemsetD32Async, etc.)
       - Device built-in types (__ppu_bfloat16 → __nv_bfloat16, etc.)
       - System header includes (hggc_*.h → cuda_*.h, etc.)
       - cuBLAS / cuRAND / cuComplex vendor prefixes (ac* → cu*)
       - Compiler macros (HGGCCC → CUDACC, etc.)
       - Note: __HGGC_ARCH__ is NOT replaced; the PPU hgcc frontend auto-defines it,
         so the cuda-compatible build keeps the native arch guard and PPU arch
         numbers (100/150) as-is.
       - Note: PTX-gating macros (CUTE_ARCH_CP_ASYNC_PPU_ENABLED,
         CUTE_ARCH_LDSM_PPU_*) are NOT renamed to SM* names. That rename is a
         coupled no-op with flash-attention, and because the affected consumers
         fall back silently (async→sync copy, no-op fence/wait), a one-sided
         rename would cause silent NaN rather than a build error.
"""

import sys
import os
import re

# ============================================================
# NAMING_REPLACEMENTS (PPU/HGGC → CUDA); order matters: longer patterns before shorter ones
# Patterns are \b-anchored so overlapping names stay independent. A few entries are
# deliberately prefix-only (no trailing \b) because the source spells longer
# variants that must convert too, e.g. acblasSgemm{,Batched,StridedBatched}.
# ============================================================
NAMING_REPLACEMENTS = [
    # === Category 1: Device built-in types (longer patterns first) ===
    (r'\b__ppu_bfloat162\b', '__nv_bfloat162'),
    (r'\b__ppu_bfloat16\b', '__nv_bfloat16'),
    (r'\bppu_bfloat162\b', 'nv_bfloat162'),
    (r'\bppu_bfloat16\b', 'nv_bfloat16'),
    (r'\b__hg_fp8_e4m3\b', '__nv_fp8_e4m3'),
    (r'\b__hg_fp8_e5m2\b', '__nv_fp8_e5m2'),

    # === Category 2: CUDA Driver / TMA symbols ===
    # (TMA / HGtensorMap* instances were removed by the open-source reform)
    (r'\bhgMemsetD32Async\b', 'cuMemsetD32Async'),
    (r'\bhgMemsetD16Async\b', 'cuMemsetD16Async'),
    (r'\bhgMemsetD8Async\b', 'cuMemsetD8Async'),
    (r'\bhgGetErrorString\b', 'cuGetErrorString'),
    (r'\bHGdeviceptr\b', 'CUdeviceptr'),
    (r'\bHGdevice\b', 'CUdevice'),
    (r'\bHGresult\b', 'CUresult'),

    # === Category 3: CUDA Runtime API functions (longer patterns first) ===
    (r'\bhggcOccupancyMaxActiveBlocksPerMultiprocessorWithFlags\b', 'cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags'),
    (r'\bhggcOccupancyMaxActiveBlocksPerMultiprocessor\b', 'cudaOccupancyMaxActiveBlocksPerMultiprocessor'),
    (r'\bhggcOccupancyMaxPotentialBlockSize\b', 'cudaOccupancyMaxPotentialBlockSize'),
    (r'\bhggcOccupancyDisableCachingOverride\b', 'cudaOccupancyDisableCachingOverride'),
    (r'\bhggcFuncAttributeMaxDynamicSharedMemorySize\b', 'cudaFuncAttributeMaxDynamicSharedMemorySize'),
    (r'\bhggcDeviceGetAttribute\b', 'cudaDeviceGetAttribute'),
    (r'\bhggcGetDeviceProperties\b', 'cudaGetDeviceProperties'),
    (r'\bhggcDevAttrMultiProcessorCount\b', 'cudaDevAttrMultiProcessorCount'),
    (r'\bhggcFuncSetAttribute\b', 'cudaFuncSetAttribute'),
    (r'\bhggcFuncAttribute\b', 'cudaFuncAttribute'),
    (r'\bhggcDeviceSynchronize\b', 'cudaDeviceSynchronize'),
    (r'\bhggcGetDeviceProp\b', 'cudaGetDeviceProp'),
    (r'\bhggcDeviceProp\b', 'cudaDeviceProp'),
    (r'\bhggcGetDevice\b', 'cudaGetDevice'),
    (r'\bhggcSetDevice\b', 'cudaSetDevice'),
    (r'\bhggcMemcpyDeviceToDevice\b', 'cudaMemcpyDeviceToDevice'),
    (r'\bhggcMemcpyDeviceToHost\b', 'cudaMemcpyDeviceToHost'),
    (r'\bhggcMemcpyHostToDevice\b', 'cudaMemcpyHostToDevice'),
    (r'\bhggcMemcpyHostToHost\b', 'cudaMemcpyHostToHost'),
    (r'\bhggcMemsetAsync\b', 'cudaMemsetAsync'),
    (r'\bhggcMemGetInfo\b', 'cudaMemGetInfo'),
    (r'\bhggcMemcpyKind\b', 'cudaMemcpyKind'),
    (r'\bhggcMemcpy\b', 'cudaMemcpy'),
    (r'\bhggcMemset\b', 'cudaMemset'),
    (r'\bhggcMalloc\b', 'cudaMalloc'),
    (r'\bhggcFree\b', 'cudaFree'),
    (r'\bhggcEventSynchronize\b', 'cudaEventSynchronize'),
    (r'\bhggcEventElapsedTime\b', 'cudaEventElapsedTime'),
    (r'\bhggcEventElapsed\b', 'cudaEventElapsed'),
    (r'\bhggcEventDestroy\b', 'cudaEventDestroy'),
    (r'\bhggcEventRecord\b', 'cudaEventRecord'),
    (r'\bhggcEventCreate\b', 'cudaEventCreate'),
    (r'\bhggcGetErrorString\b', 'cudaGetErrorString'),
    (r'\bhggcGetErrorName\b', 'cudaGetErrorName'),
    (r'\bhggcPeekAtLastError\b', 'cudaPeekAtLastError'),
    (r'\bhggcGetLastError\b', 'cudaGetLastError'),

    # === Category 4: CUDA Runtime types ===
    (r'\bhggcStream_t\b', 'cudaStream_t'),
    (r'\bhggcSuccess\b', 'cudaSuccess'),
    (r'\bhggcError_t\b', 'cudaError_t'),
    (r'\bhggcError\b', 'cudaError'),
    (r'\bhggcEvent_t\b', 'cudaEvent_t'),
    (r'\bhggcDataType_t\b', 'cudaDataType_t'),
    (r'\bhggcDataType\b', 'cudaDataType'),

    # === Category 5: System header includes (hggc → cuda; omit <> to handle both quote styles) ===
    (r'\bhggc_runtime_api\.h\b', 'cuda_runtime_api.h'),
    (r'\bhggc_runtime\.h\b', 'cuda_runtime.h'),
    (r'\bhggc_fp16\.h\b', 'cuda_fp16.h'),
    (r'\bhggc_fp8\.h\b', 'cuda_fp8.h'),
    (r'\bhggc_bf16\.h\b', 'cuda_bf16.h'),
    (r'\bhggc_vector_types\.h\b', 'vector_types.h'),
    (r'\bacComplex\.h\b', 'cuComplex.h'),

    # === ac* complex type → cu* mapping (long patterns first) ===
    (r'\bmake_acDoubleComplex\b', 'make_cuDoubleComplex'),
    (r'\bmake_acFloatComplex\b', 'make_cuFloatComplex'),
    (r'\bacDoubleComplex\b', 'cuDoubleComplex'),
    (r'\bacFloatComplex\b', 'cuFloatComplex'),
    (r'\bacCrealf\b', 'cuCrealf'),
    (r'\bacCimagf\b', 'cuCimagf'),
    (r'\bacCreal\b', 'cuCreal'),
    (r'\bacCimag\b', 'cuCimag'),
    (r'\bhggc\.h\b', 'cuda.h'),
    (r'\bhggc/std/', 'cuda/std/'),

    # cuBLAS status enums (ACBLAS → CUBLAS)
    (r'\bACBLAS_GEMM_DEFAULT_TENSOR_OP\b', 'CUBLAS_GEMM_DEFAULT_TENSOR_OP'),
    (r'\bACBLAS_STATUS_NOT_INITIALIZED\b', 'CUBLAS_STATUS_NOT_INITIALIZED'),
    (r'\bACBLAS_STATUS_EXECUTION_FAILED\b', 'CUBLAS_STATUS_EXECUTION_FAILED'),
    (r'\bACBLAS_STATUS_ALLOC_FAILED\b', 'CUBLAS_STATUS_ALLOC_FAILED'),
    (r'\bACBLAS_STATUS_INVALID_VALUE\b', 'CUBLAS_STATUS_INVALID_VALUE'),
    (r'\bACBLAS_STATUS_ARCH_MISMATCH\b', 'CUBLAS_STATUS_ARCH_MISMATCH'),
    (r'\bACBLAS_STATUS_MAPPING_ERROR\b', 'CUBLAS_STATUS_MAPPING_ERROR'),
    (r'\bACBLAS_STATUS_INTERNAL_ERROR\b', 'CUBLAS_STATUS_INTERNAL_ERROR'),
    (r'\bACBLAS_STATUS_NOT_SUPPORTED\b', 'CUBLAS_STATUS_NOT_SUPPORTED'),
    (r'\bACBLAS_STATUS_LICENSE_ERROR\b', 'CUBLAS_STATUS_LICENSE_ERROR'),
    (r'\bACBLAS_STATUS_SUCCESS\b', 'CUBLAS_STATUS_SUCCESS'),
    (r'\bACBLAS_ERROR\b', 'CUBLAS_ERROR'),

    # cuBLAS types & functions (acblas* → cublas*)
    # The five *gemm entries are prefix-only: the source also spells
    # acblasSgemmBatched / acblasSgemmStridedBatched, which must convert too.
    (r'\bacblasOperation_t\b', 'cublasOperation_t'),
    (r'\bacblasStatus_t\b', 'cublasStatus_t'),
    (r'\bacblasHandle_t\b', 'cublasHandle_t'),
    (r'\bacblasGemmEx\b', 'cublasGemmEx'),
    (r'\bacblasSgemm', 'cublasSgemm'),
    (r'\bacblasDgemm', 'cublasDgemm'),
    (r'\bacblasHgemm', 'cublasHgemm'),
    (r'\bacblasCgemm', 'cublasCgemm'),
    (r'\bacblasZgemm', 'cublasZgemm'),

    # cuRAND types & functions (acrand* → curand*)
    (r'\bacrandStateXORWOW_t\b', 'curandStateXORWOW_t'),
    (r'\bacrandStateXORWOW\b', 'curandStateXORWOW'),
    (r'\bacrand_uniform_double\b', 'curand_uniform_double'),
    (r'\bacrand_normal_double\b', 'curand_normal_double'),
    (r'\bacrand_uniform\b', 'curand_uniform'),
    (r'\bacrand_normal\b', 'curand_normal'),
    (r'\bacrandState_t\b', 'curandState_t'),
    (r'\bacrandState\b', 'curandState'),
    (r'\bacrand_init\b', 'curand_init'),

    # === M4: CUTLASS internal macros + compiler macro mapping ===
    # Host adapter: type rename → no change (open-source name kept)

    # Barrier + Clang (internal macros, not converted)

    # Platform macros (__HGGC_STD_ is prefix-only, so the explicit names come first)
    (r'\b__HGGC_STD_XYZ\b', '__NV_STD_XYZ'),
    (r'\b__HGGC_STD_MIN\b', '__NV_STD_MIN'),
    (r'\b__HGGC_STD_MAX\b', '__NV_STD_MAX'),
    (r'\b__HGGC_STD_', '__NV_STD_'),

    # CUDA API constants
    (r'\bHGGC_SUCCESS\b', 'CUDA_SUCCESS'),
    (r'\bhggcErrorUnknown\b', 'cudaErrorUnknown'),
    (r'\bHGGC_C_64F\b', 'CUDA_C_64F'),
    (r'\bHGGC_C_32F\b', 'CUDA_C_32F'),
    (r'\bHGGC_C_16F\b', 'CUDA_C_16F'),
    (r'\bHGGC_R_64F\b', 'CUDA_R_64F'),
    (r'\bHGGC_R_32I\b', 'CUDA_R_32I'),
    (r'\bHGGC_R_32F\b', 'CUDA_R_32F'),
    (r'\bHGGC_R_16F\b', 'CUDA_R_16F'),
    (r'\bHGGC_R_8I\b', 'CUDA_R_8I'),

    # Compiler/runtime version macros (HGGC → CUDA, longer patterns first)
    (r'\b__HGGC_NO_HALF2_OPERATORS__\b', '__CUDA_NO_HALF2_OPERATORS__'),
    (r'\b__HGGC_NO_HALF_OPERATORS__\b', '__CUDA_NO_HALF_OPERATORS__'),
    (r'\bHGGCRT_VERSION\b', 'CUDART_VERSION'),
    (r'\b__HGGCCC_RTC__\b', '__CUDACC_RTC__'),
    (r'\b__HGGCCC_VER_MAJOR__\b', '__CUDACC_VER_MAJOR__'),
    (r'\b__HGGCCC_VER_MINOR__\b', '__CUDACC_VER_MINOR__'),
    (r'\b__HGGCCC__\b', '__CUDACC__'),

    # Clang CUDA macro
    (r'\b__HGGC__\b', '__CUDA__'),

]

def apply_naming_replacements(content):
    """Apply naming conversion rules (PPU/HGGC → CUDA) to file content."""
    for pattern, replacement in NAMING_REPLACEMENTS:
        content = re.sub(pattern, replacement, content)
    return content


def process_directory(include_dir):
    """Walk the include/ directory and apply naming conversion to all C/C++ files."""
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

    # Handle file renames (none — open-source-reform file names are preserved)
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
