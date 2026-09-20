#!/usr/bin/env python3
"""
cuda_compat.py - Convert PPU original code to CUDA-compatible code.

Usage:
    python3 cuda_compat.py [REPO_DIR] [--dry-run] [--verbose] [--reference-dir PATH]

This script converts the PPU original DeepGEMM codebase to a CUDA-compatible version.
It operates ONLY on the DeepGEMM repository directory (excludes third-party/).
No git or network dependency required.
"""

import argparse
import os
import re
import stat
import sys
from pathlib import Path
from typing import List, Tuple, Dict

# =============================================================================
# Configuration
# =============================================================================

# Default repo directory (relative to this script)
DEFAULT_REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DeepGemm')

# Directories/files to EXCLUDE from processing
EXCLUDE_PATTERNS = [
    'third-party/',
    '.git/',
    '.eggs/',
    'build/',
    'dist/',
    'deep_gemm.egg-info/',
    '__pycache__/',
    '*.so',
    '*.pyc',
    '*.egg',
]
# Always exclude this script itself (by its actual filename), so running it from
# inside the repo never treats the script as a source file to be converted.
# NOTE: this only protects the *running* script; if you keep a backup copy or
# another convert script inside the repo, add its name here explicitly.
_self = os.path.basename(__file__)
if _self not in EXCLUDE_PATTERNS:
    EXCLUDE_PATTERNS.append(_self)

# File extensions to process
TARGET_EXTENSIONS = {'.py', '.hpp', '.cuh', '.cu', '.cpp', '.h', '.sh', '.md'}

# =============================================================================
# LEVEL 1: Rule-based text replacements (ordered by specificity, longest first)
# Format: (ppu/hggc pattern, cuda/nv equivalent)
# =============================================================================

# --- C/C++ Header includes ---
INCLUDE_REPLACEMENTS = [
    # Headers
    ('<hggc_runtime.h>', '<cuda_runtime.h>'),
    ('<hggc.h>', '<cuda.h>'),
    ('<hgrtc.h>', '<nvrtc.h>'),
    ('<hggc_fp8.h>', '<cuda_fp8.h>'),
    ('<hggc_fp16.h>', '<cuda_fp16.h>'),
    ('<hggc_bf16.h>', '<cuda_bf16.h>'),
    ('<hgtx3/hgToolsExt.h>', '<nvtx3/nvToolsExt.h>'),
    ('<hggc_pipeline.h>', '<cuda_pipeline.h>'),
    ('<hggc/std/cstdint>', '<cuda/std/cstdint>'),
    # NOTE: '#include <torch/torch.h>' → ATen/cuda/CUDAContext.h is NOT a global rule.
    # common_fp4.hpp legitimately uses torch/torch.h in both versions.
    # File-specific handling in device_runtime.hpp if needed.

    # NOTE: '<acblasLt.h>' needs no entry -- BLAS_REPLACEMENTS' 'acblas' -> 'cublas' covers it.
]

# --- acBLAS/acBLASLt -> cuBLAS/cuBLASLt ---
BLAS_REPLACEMENTS = [
    ('ACBLAS', 'CUBLAS'),
    ('acBLAS', 'cuBLAS'),
    ('Acblas', 'Cublas'),
    ('acblas', 'cublas'),
]

# --- Macro definitions and checks ---
MACRO_REPLACEMENTS = [
    # Version macros
    ('HGGCRT_VERSION', 'CUDART_VERSION'),
    ('HGGC_VERSION', 'CUDA_VERSION'),
    # NVRTC defines in code strings
    ('BF16_HGRTC', 'BF16_NVRTC'),
    ('FP8_HGRTC', 'FP8_NVRTC'),
    ('INT8_HGRTC', 'INT8_NVRTC'),
]

# --- Type replacements (C/C++) ---
TYPE_REPLACEMENTS = [
    # Cutlass host adapter type: NOT using simple string replace here because
    # 'HostAdapter' appears inside 'CudaHostAdapter' causing double-replacement.
    # Handled via REGEX_REPLACEMENTS with negative lookbehind instead.
    # Driver API types
    ('HGtensorMapDataType', 'CUtensorMapDataType'),
    ('HGtensorMapSwizzle', 'CUtensorMapSwizzle'),
    ('HGtensorMap', 'CUtensorMap'),
    ('HGmodule', 'CUmodule'),
    ('HGfunction', 'CUfunction'),
    ('HGresult', 'CUresult'),
    ('HGlaunchConfig', 'CUlaunchConfig'),
    ('HGlaunchAttribute', 'CUlaunchAttribute'),
    # Driver API library/kernel handles (handle.hpp's DG_JIT_USE_LIBRARY_ENUM_KERNELS path)
    ('HGlibrary', 'CUlibrary'),
    ('HGkernel', 'CUkernel'),
    ('hg_kernel', 'cu_kernel'),
    # Runtime API types
    ('hggcDeviceProp', 'cudaDeviceProp'),
    ('hggcStream_t', 'cudaStream_t'),
    ('hggcError_t', 'cudaError_t'),
    ('hggcLibrary_t', 'cudaLibrary_t'),
    ('hggcKernel_t', 'cudaKernel_t'),
    ('hggcLaunchConfig_t', 'cudaLaunchConfig_t'),
    ('hggcLaunchAttribute', 'cudaLaunchAttribute'),
    ('hggcStreamCaptureStatus', 'cudaStreamCaptureStatus'),
    # Integer types
    ('hguint64_t', 'cuuint64_t'),
    ('hguint32_t', 'cuuint32_t'),
    # Device type intrinsics
    ('__ppu_bfloat16', '__nv_bfloat16'),
    ('__hg_fp8_e4m3', '__nv_fp8_e4m3'),
    # Runtime API types (additional)
    ('hggcFuncAttributes', 'cudaFuncAttributes'),
    ('hggcFuncGetAttributes', 'cudaFuncGetAttributes'),
    ('hggcOccupancyMaxActiveBlocksPerMultiprocessor', 'cudaOccupancyMaxActiveBlocksPerMultiprocessor'),
    # BLAS scalar-type plumbing (smxx_acblaslt.hpp); `hggc_type_` keeps the trailing `_`
    ('hggcDataType', 'cudaDataType'),
    ('hggc_type_', 'cuda_type_'),
]

# --- Enum/constant replacements ---
ENUM_REPLACEMENTS = [
    # TensorMap data types
    ('HG_TENSOR_MAP_DATA_TYPE_TFLOAT32', 'CU_TENSOR_MAP_DATA_TYPE_TFLOAT32'),
    ('HG_TENSOR_MAP_DATA_TYPE_INT32', 'CU_TENSOR_MAP_DATA_TYPE_INT32'),
    ('HG_TENSOR_MAP_DATA_TYPE_FLOAT32', 'CU_TENSOR_MAP_DATA_TYPE_FLOAT32'),
    ('HG_TENSOR_MAP_DATA_TYPE_BFLOAT16', 'CU_TENSOR_MAP_DATA_TYPE_BFLOAT16'),
    ('HG_TENSOR_MAP_DATA_TYPE_UINT8', 'CU_TENSOR_MAP_DATA_TYPE_UINT8'),
    # TensorMap swizzle
    ('HG_TENSOR_MAP_SWIZZLE_128B_ATOM_32B', 'CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B'),
    ('HG_TENSOR_MAP_SWIZZLE_NONE', 'CU_TENSOR_MAP_SWIZZLE_NONE'),
    ('HG_TENSOR_MAP_SWIZZLE_32B', 'CU_TENSOR_MAP_SWIZZLE_32B'),
    ('HG_TENSOR_MAP_SWIZZLE_64B', 'CU_TENSOR_MAP_SWIZZLE_64B'),
    ('HG_TENSOR_MAP_SWIZZLE_128B', 'CU_TENSOR_MAP_SWIZZLE_128B'),
    # TensorMap misc
    ('HG_TENSOR_MAP_INTERLEAVE_NONE', 'CU_TENSOR_MAP_INTERLEAVE_NONE'),
    ('HG_TENSOR_MAP_L2_PROMOTION_L2_256B', 'CU_TENSOR_MAP_L2_PROMOTION_L2_256B'),
    ('HG_TENSOR_MAP_FLOAT_OOB_FILL_NONE', 'CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE'),
    # Function attributes
    ('HG_FUNC_ATTRIBUTE_NUM_REGS', 'CU_FUNC_ATTRIBUTE_NUM_REGS'),
    ('HG_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES', 'CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES'),
    ('HG_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES', 'CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES'),
    # Error codes
    ('HGGC_SUCCESS', 'CUDA_SUCCESS'),
    ('HGGC_ERROR_DEINITIALIZED', 'CUDA_ERROR_DEINITIALIZED'),
    ('HGRTC_SUCCESS', 'NVRTC_SUCCESS'),
    ('hggcSuccess', 'cudaSuccess'),
    ('hggcErrorHggcrtUnloading', 'cudaErrorCudartUnloading'),
    # NVTX enums
    ('HGTX_VERSION', 'NVTX_VERSION'),
    ('HGTX_EVENT_ATTRIB_STRUCT_SIZE', 'NVTX_EVENT_ATTRIB_STRUCT_SIZE'),
    ('HGTX_MESSAGE_TYPE_ASCII', 'NVTX_MESSAGE_TYPE_ASCII'),
    # Device attribute enums (for ppu_occ_model.cuh)
    ('hggcDevAttrMaxSharedMemoryPerMultiprocessor', 'cudaDevAttrMaxSharedMemoryPerMultiprocessor'),
    ('hggcDevAttrMaxThreadsPerMultiProcessor', 'cudaDevAttrMaxThreadsPerMultiProcessor'),
    ('hggcDevAttrMaxRegistersPerMultiprocessor', 'cudaDevAttrMaxRegistersPerMultiprocessor'),
    ('hggcDevAttrMaxThreadsPerBlock', 'cudaDevAttrMaxThreadsPerBlock'),
    ('hggcDevAttrWarpSize', 'cudaDevAttrWarpSize'),
    # Data-type enums (HGGC_R_32F, HGGC_R_16BF, ...).
    ('HGGC_R_', 'CUDA_R_'),
]

# --- Function/API name replacements ---
FUNC_REPLACEMENTS = [
    # Driver API lazy wrappers
    ('lazy_hgGetErrorName', 'lazy_cuGetErrorName'),
    ('lazy_hgGetErrorString', 'lazy_cuGetErrorString'),
    ('lazy_hgFuncSetAttribute', 'lazy_cuFuncSetAttribute'),
    ('lazy_hgModuleLoad', 'lazy_cuModuleLoad'),
    ('lazy_hgModuleUnload', 'lazy_cuModuleUnload'),
    ('lazy_hgModuleGetFunction', 'lazy_cuModuleGetFunction'),
    ('lazy_hgLaunchKernelEx', 'lazy_cuLaunchKernelEx'),
    ('lazy_hgTensorMapEncodeTiled', 'lazy_cuTensorMapEncodeTiled'),
    # Driver API direct calls (bare, without lazy_ prefix — used in DECL_LAZY macros)
    ('hgGetErrorName', 'cuGetErrorName'),
    ('hgGetErrorString', 'cuGetErrorString'),
    ('hgFuncSetAttribute', 'cuFuncSetAttribute'),
    ('hgModuleLoad', 'cuModuleLoad'),
    ('hgModuleUnload', 'cuModuleUnload'),
    ('hgLaunchKernelEx', 'cuLaunchKernelEx'),
    ('hgTensorMapEncodeTiled', 'cuTensorMapEncodeTiled'),
    # Driver API direct calls (used in some files without lazy)
    ('hgFuncGetAttribute', 'cuFuncGetAttribute'),
    ('hgOccupancyMaxActiveBlocksPerMultiprocessor', 'cuOccupancyMaxActiveBlocksPerMultiprocessor'),
    ('hgModuleLoadData', 'cuModuleLoadData'),
    ('hgModuleGetFunction', 'cuModuleGetFunction'),
    # Driver API library/kernel enumeration (handle.hpp); the bare names also fix `lazy_*`
    ('hgLibraryLoadFromFile', 'cuLibraryLoadFromFile'),
    ('hgLibraryUnload', 'cuLibraryUnload'),
    ('hgLibraryGetKernelCount', 'cuLibraryGetKernelCount'),
    ('hgLibraryEnumerateKernels', 'cuLibraryEnumerateKernels'),
    ('hgKernelGetFunction', 'cuKernelGetFunction'),
    # Runtime API
    ('hggcGetDevice', 'cudaGetDevice'),
    ('hggcGetDeviceProperties', 'cudaGetDeviceProperties'),
    ('hggcFuncSetAttribute', 'cudaFuncSetAttribute'),
    ('hggcFuncAttributeMaxDynamicSharedMemorySize', 'cudaFuncAttributeMaxDynamicSharedMemorySize'),
    ('hggcLibraryLoadFromFile', 'cudaLibraryLoadFromFile'),
    ('hggcLibraryGetKernel', 'cudaLibraryGetKernel'),
    ('hggcLibraryUnload', 'cudaLibraryUnload'),
    ('hggcLaunchKernelExC', 'cudaLaunchKernelExC'),
    ('hggcDeviceSynchronize', 'cudaDeviceSynchronize'),
    ('hggcStreamSynchronize', 'cudaStreamSynchronize'),
    ('hggcStreamIsCapturing', 'cudaStreamIsCapturing'),
    ('hggcGetLastError', 'cudaGetLastError'),
    ('hggcDeviceGetAttribute', 'cudaDeviceGetAttribute'),
    ('hggcGetErrorName', 'cudaGetErrorName'),
    ('hggcGetErrorString', 'cudaGetErrorString'),
    ('hggcMemcpyAsync', 'cudaMemcpyAsync'),
    ('hggcMemcpyDeviceToHost', 'cudaMemcpyDeviceToHost'),
    ('hggcMemsetAsync', 'cudaMemsetAsync'),
    ('hggcStreamCaptureStatusNone', 'cudaStreamCaptureStatusNone'),
    # Path/binary variable names (cross-file)
    ('hgbin_path', 'cubin_path'),
    ('hgobjdump_path', 'cuobjdump_path'),
    # Library name
    ('libhggc.so', 'libcuda.so.1'),
    # NVRTC API
    ('hgrtcVersion', 'nvrtcVersion'),
    ('hgrtcCreateProgram', 'nvrtcCreateProgram'),
    ('hgrtcCompileProgram', 'nvrtcCompileProgram'),
    ('hgrtcGetProgramLogSize', 'nvrtcGetProgramLogSize'),
    ('hgrtcGetProgramLog', 'nvrtcGetProgramLog'),
    ('hgrtcGetHGBINSize', 'nvrtcGetCUBINSize'),
    ('hgrtcGetHGBIN', 'nvrtcGetCUBIN'),
    ('hgrtcDestroyProgram', 'nvrtcDestroyProgram'),
    ('hgrtcGetErrorString', 'nvrtcGetErrorString'),
    # Binary/file naming
    ('kernel.hgbin', 'kernel.cubin'),
    ('Loading HGBIN:', 'Loading CUBIN:'),
    ('HGGC driver API', 'CUDA driver API'),
    # NVRTC types
    ('hgrtcProgram', 'nvrtcProgram'),
    # NVTX functions
    ('hgtxEventAttributes_t', 'nvtxEventAttributes_t'),
    ('hgtxDomainRangePushEx', 'nvtxDomainRangePushEx'),
    ('hgtxDomainRangePop', 'nvtxDomainRangePop'),
    ('hgtxDomainCreateA', 'nvtxDomainCreateA'),
    ('hgtxDomainDestroy', 'nvtxDomainDestroy'),
    ('hgtxDomainHandle_t', 'nvtxDomainHandle_t'),
]

# --- Python-specific replacements ---
PYTHON_REPLACEMENTS = [
    # Compiler naming
    ('get_hgcc_compiler', 'get_nvcc_compiler'),
    ("DG_JIT_HGCC_COMPILER", "DG_JIT_NVCC_COMPILER"),
    ('DG_JIT_USE_HGRTC', 'DG_JIT_USE_NVRTC'),
    ("DG_JIT_PRINT_HGCC_COMMAND", "DG_JIT_PRINT_NVCC_COMMAND"),
    ("DG_HGCC_OVERRIDE_CPP_STANDARD", "DG_NVCC_OVERRIDE_CPP_STANDARD"),
    # PPU SDK env vars -> CUDA_HOME
    ('PPU_SDK', 'CUDA_HOME'),
    ('PPU_HOME', 'CUDA_HOME'),
    # Library names
    ("'hggc'", "'cuda'"),
    ("'hggcrt1'", "'cudart'"),
    ("'hgrtc'", "'nvrtc'"),
]

# =============================================================================
# LEVEL 2: File-level operations
# =============================================================================

# Files to handle: if .cpp exists and .cu does not, rename; if both exist, remove .cpp
FILE_RENAMES = [
    ('csrc/python_api.cpp', 'csrc/python_api.cu'),
    ('csrc/jit_kernels/impls/smxx_acblaslt.hpp', 'csrc/jit_kernels/impls/smxx_cublaslt.hpp'),
]

# Files to restore (were deleted in ppu-original; provide content to recreate)
# CMakeLists.txt was deleted - will be restored separately

# =============================================================================
# LEVEL 3: Complex / "hardcoded" replacements that cannot be simple string subs
# These are specific multi-line or contextual changes.
# =============================================================================

# Regex-based replacements: (pattern, replacement, file_glob_or_None)
REGEX_REPLACEMENTS: List[Tuple[str, str, str]] = [

    # In impls/fp4_gemm.hpp ONLY: expand launch_kernel to direct API call
    # Other impls files (bf16, fp8, int8) keep launch_kernel as-is in the target
    (r'DG_CUDA_UNIFIED_CHECK\(launch_kernel\(kernel, configs, args\.kernel_params\)\);',
     'void* ptr_args[] = {(void*)&args.kernel_params};\n        DG_CUDA_UNIFIED_CHECK(lazy_cuLaunchKernelEx(&configs, kernel, ptr_args, nullptr));', 'fp4_gemm.hpp'),


    # --- utils_rtc.cuh ---
    # One generic rule for both `#if defined(__HGGC__)` regions (atomic_add_release_global
    # and computeBlockInfoKernel): keep the body, drop the guard. nvcc does not define
    # __HGGC__, so leaving the guard in place would delete the code from the product.
    # The body pattern is LINE-BOUNDED and refuses to cross any other preprocessor
    # directive, so it cannot mis-pair one region's #if with another region's #endif --
    # the failure mode of the DOTALL `.*?` patterns this replaces. It also does not
    # depend on any comment text or on blank-line counts.
    (r'(?m)^#if defined\(__HGGC__\)\n((?:(?!^#\s*(?:if|ifdef|ifndef|else|elif|endif)\b)[^\n]*\n)*)^#endif  // __HGGC__\n',
     r'\1', 'utils_rtc.cuh'),

    # NOTE: next_power_of_two needs no rule -- in utils_rtc.cuh it sits outside every
    # guard and is valid for both backends, so it simply survives conversion.

    # --- profiling_interface.cuh ---
    # The DG_USE_HGTX guards used to be stripped here by 9 positional regexes that
    # depended on \s+ indentation and on the HGTX->NVTX rename having already run.
    # DG_USE_HGTX was never defined anywhere, so the guarded code was dead on the PPU
    # side while the CUDA side needs the nvtx calls unguarded; the PPU source now simply
    # has no guards, and conversion is pure token renaming. Only this include drop remains.
    (r'#include <cuda_runtime.h>\n', '', 'profiling_interface.cuh'),

    # --- scheduler_cutlass3.cuh: resolve the #if defined(__HGGC__) guards ---
    # Remove orphaned #endif after pragma lines (left over from #ifdef __clang__ removal)
    # Remove #if defined(__HGGC__) and its matching #endif (keeping content between)
    # NOTE: the __clang__ and USE_HGGC guards are gone from the PPU source itself.
    # hgcc does not predefine __clang__ (so those pragmas never applied), and USE_HGGC
    # is passed by every compile path including hgrtc, so both guards were inert.
    # The __HGGC__ guards below MUST stay: csrc/python_api.cpp is a host TU built by
    # g++, which does not define __HGGC__, and the guarded code uses threadIdx /
    # blockDim / __shfl_sync / __syncthreads.
    # One line-bounded rule for all four `#if defined(__HGGC__)` regions: keep the body,
    # drop the guard (nvcc does not define __HGGC__, so the guard would delete the code).
    # It replaces three patterns that split the work by whether the #if happened to be
    # followed by a '// --- Device-only ...' comment -- rewording that comment used to
    # leave a stray #if while its #endif was still removed. The body pattern refuses to
    # cross any other preprocessor directive, so regions can never be mis-paired.
    (r'(?m)^#if defined\(__HGGC__\)\n((?:(?!^#\s*(?:if|ifdef|ifndef|else|elif|endif)\b)[^\n]*\n)*)^#endif  // defined\(__HGGC__\)\n',
     r'\1', 'scheduler_cutlass3.cuh'),
    # Remove standalone comment about host-callable methods added in ppu-original
    # NOTE: two blank-line-only rules were removed here. They normalised the number
    # of empty lines around '};' and did not change a single token, so any
    # reformatting of the source silently disabled them. The product now keeps the
    # source's blank lines, which cannot affect compilation.
    # Remove extra blank lines between closing } and }; near the Dynamic Tile comment
    # Remove triple newline before #pragma clang diagnostic pop
]


# Specific whole-line or block replacements per file
# Format: {relative_path: [(old_text, new_text), ...]}
FILE_SPECIFIC_REPLACEMENTS: Dict[str, List[Tuple[str, str]]] = {
    # --- csrc/jit/compiler.hpp (ACTIVE) ---
    # Pure token/phrase renames only. All genuinely divergent multi-line regions are
    # handled by FILE_SPECIFIC_BLOCK_REPLACEMENTS (anchor-delimited). This list is
    # robust to incidental edits (whitespace/comments/reordering) around these tokens.
    'csrc/jit/compiler.hpp': [
        # Includes (file-specific; differ from generic hggc.h -> cuda.h mapping)
        ('#include <hggc_runtime_api.h>', '#include <ATen/cuda/CUDAContext.h>'),
        ('#include <hggc.h>', '#include <cuda_runtime.h>'),
        # The offline PPU compiler rejects both --diag-suppress and --ptxas-options, so the
        # base ctor seeds neither; both are re-injected here for the CUDA build. -lineinfo
        # IS accepted by the PPU compiler and therefore stays on both sides.
        ('flags = fmt::format("-std=c++{} ",',
         'flags = fmt::format("-std=c++{} --diag-suppress=39,174,177,940 ",'),
        # Anchored on the 8-space, non-template LINEINFO if-line (the RtcOptions copy uses
        # get_env<int>(...) with 12 spaces, so it is not touched).
        ('        if (get_env("DG_JIT_WITH_LINEINFO", 0))\n',
         '        if (get_env("DG_JIT_DEBUG", 0) or get_env("DG_JIT_PTXAS_VERBOSE", 0) or get_env("DG_JIT_PTXAS_CHECK", 0))\n            flags += " --ptxas-options=--verbose,--warn-on-local-memory-usage";\n        if (get_env("DG_JIT_WITH_LINEINFO", 0))\n'),
        # The PPU-only defines collapse to a bare separator (the base flags do not always
        # end with a space, so the space must be kept).
        (' -DUSE_HGGC -DUSE_CLANG -DUSE_ACWRAPPER ', ' '),
        # Member / parameter / local variable renames (longer patterns first)
        ('hgcc_path', 'nvcc_path'),
        ('hgcc_extra_flags', 'nvcc_flags'),
        # RTC include lists: the CUDA side needs extra toolkit paths, so these are
        # value rules rather than renames. They must precede the sdk_include* renames.
        ('includes_insert({sdk_include, sdk_include_cccl});',
         'includes_insert({cuda_home, cuda_home1, cuda_home1 + "/cuda/std"});'),
        ('includes_insert({sdk_include});',
         'includes_insert({cuda_home, cuda_home + "/cuda/std", cuda_home + "/../targets/x86_64-linux/include/thrust/system/cuda"});'),
        ('sdk_include_cccl', 'cuda_home1'),
        ('sdk_include', 'cuda_home'),
        # HGRTC ctor: signature + restore NVRTC version assertion
        ('signature = fmt::format("HGRTC{}.{}", major, minor);',
         'signature = fmt::format("NVRTC{}.{}", major, minor);\n        DG_HOST_ASSERT((major > 12 or (major == 12 and minor >= 3)) and "NVRTC version should be >= 12.3");'),
        # RtcOptions opts init: -DUSE_HGGC -> -D__CUDACC__
        ('            "-DUSE_HGGC",', '            // "-D__CUDACC_RTC__",\n            "-D__CUDACC__",'),
        # Compiler path + version/signature (pure token; no structural block needed)
        ('"bin" / "hgcc"', '"bin" / "nvcc"'),
        ('DG_JIT_HGCC_COMPILER', 'DG_JIT_NVCC_COMPILER'),
        ('signature = fmt::format("HGCC{}", get_hgcc_version());',
         'const auto& [nvcc_major, nvcc_minor] = get_nvcc_version();\n        signature = fmt::format("NVCC{}.{}", nvcc_major, nvcc_minor);'),
        # Flag-value translation (PPU offline hgcc flags -> nvcc equivalents). Makes the
        # token-converted output produce a flag SET equivalent to the hand-written
        # cuda-compat version, while cuda-free keeps its readable per-section structure.
        ('-arch=ppu_15 ', '-gencode=arch=compute_89,code=sm_89 '),
        ('-arch=ppu_10 ', '-gencode=arch=compute_80a,code=sm_80a '),
        ('-hgbin -ftemplate-depth=8192 -O3 -DNDEBUG ', '-cubin --expt-relaxed-constexpr --expt-extended-lambda '),
        ('-Xcompiler -fPIC ', ''),
        ('-Xcompiler -Wno-deprecated-declarations -Xcompiler -Wno-abi ', '-Xcompiler -O3,-Wno-deprecated-declarations,-Wno-abi '),
        # PPU LLVM backend flag prefix
        ('-Xllvm', '-mllvm'),
    ],

    # --- csrc/jit/device_runtime.hpp ---
    'csrc/jit/device_runtime.hpp': [
        ('#include <hggc_runtime_api.h>', '#include <cuda_runtime.h>'),
        ('#include <torch/torch.h>', '#include <ATen/cuda/CUDAContext.h>'),
    ],

    # --- deep_gemm/include/deep_gemm/impls/w4a16_gemm_cutlass3.cuh ---
    'deep_gemm/include/deep_gemm/impls/w4a16_gemm_cutlass3.cuh': [
        # Add #pragma clang diagnostic ignored after push
        ('#pragma clang diagnostic push\n#pragma clang diagnostic ignored "-Wunknown-attributes"',
         '#pragma clang diagnostic push\n#pragma clang diagnostic ignored "-Wunknown-attributes"\n#pragma clang diagnostic ignored "-Wcuda-compat"'),
    ],

    # --- setup.py ---
    'setup.py': [
        # The SDK path block is dropped. Split in two so removing the *code* is not
        # coupled to the *comment* text: if someone rewords the comment, only the
        # (harmless) comment rule stops matching, while verify_required still guards
        # the code rules. Neither depends on surrounding blank lines.
        ("ppu_sdk = os.environ.get('PPU_SDK', '/usr/local/PPU_SDK')\n"
         "ppu_include = os.path.join(ppu_sdk, 'targets', 'x86_64-linux', 'include')\n\n",
         ""),
        ("# PPU SDK path \u2014 provides hggc headers for actlize\n", ""),
        ("sources = ['csrc/python_api.cpp']", "sources = ['csrc/python_api.cu']"),
        # Fix include dirs
        ("    ppu_include,", "    f'{CUDA_HOME}/include',"),
        # NOTE: build_libraries needs no rule -- the generic PYTHON_REPLACEMENTS
        # tokens ('hggc'->'cuda', 'hggcrt1'->'cudart', 'hgrtc'->'nvrtc') already
        # produce the exact target text.
        # Fix library dirs
        ("    os.path.join(ppu_sdk, 'lib'),",
         "    f'{CUDA_HOME}/lib64',\n    f'{CUDA_HOME}/lib64/stub'"),
        # Fix compiler flags: remove -DUSE_HGGC from cxx_args
        ('extra_cxx_args = ["-O3", "-std=c++17", "-DUSE_HGGC"]',
         'extra_cxx_args = ["-O3", "-std=c++17"]'),
        # Fix nvcc args: rename and remove -DUSE_HGGC
        ('extra_hgcc_args = ["-O3", "-std=c++17", "--use_fast_math", "-DUSE_HGGC"]',
         'extra_nvcc_args = ["-O3", "-std=c++17", "--use_fast_math"]'),
        ('extra_hgcc_args.append', 'extra_nvcc_args.append'),
        ('"hgcc": extra_hgcc_args,', '"nvcc": extra_nvcc_args,'),
        # Extension type
        ("CppExtension(name='deep_gemm.deep_gemm_cpp',", "CUDAExtension(name='deep_gemm.deep_gemm_cpp',"),
        ("from torch.utils.cpp_extension import CppExtension, BuildExtension",
         "from torch.utils.cpp_extension import CppExtension, CUDA_HOME, CUDAExtension, BuildExtension"),
    ],

    # --- deep_gemm/__init__.py ---
    'deep_gemm/__init__.py': [
        # The PPU source keeps deep_gemm_cpp.init() at its final position and derives the
        # SDK root in a single expression, so only this import swap is needed. It must run
        # before the generic PPU_HOME -> CUDA_HOME token (which would otherwise rewrite the
        # definition into a self-shadowing CUDA_HOME assignment); file-specific rules do.
        ("# SDK root: env PPU_SDK > env PPU_HOME > default\nPPU_HOME = os.environ.get('PPU_SDK') or os.environ.get('PPU_HOME') or '/usr/local/PPU_SDK'",
         'from torch.version import cuda as cuda_version\nfrom packaging import version\nfrom torch.utils.cpp_extension import CUDA_HOME'),
    ],

    # --- deep_gemm/jit/compiler.py (token-only; structural part is a block) ---
    'deep_gemm/jit/compiler.py': [
        # CUDA_HOME import needed by the converted get_nvcc_compiler()
        ('import torch\nfrom typing import Tuple',
         'import torch\nfrom torch.utils.cpp_extension import CUDA_HOME\nfrom typing import Tuple'),
        # Flag-value translation (PPU -> nvcc). Equivalent flag SET, per-section structure kept.
        ("'-DUSE_HGGC', '-DUSE_CLANG', '-DUSE_ACWRAPPER',",
         "'--expt-relaxed-constexpr', '--expt-extended-lambda', '--diag-suppress=39,174,177,940',"),
        ("'-arch=ppu_15'", "'-gencode=arch=compute_89,code=sm_89'"),
        ("'-arch=ppu_10'", "'-gencode=arch=compute_80a,code=sm_80a'"),
        ("['-ftemplate-depth=8192', '-O3', '-DNDEBUG']", "['-O3']"),
        ("['-Xcompiler', '-fPIC', '-Xcompiler', '-Wno-deprecated-declarations', '-Xcompiler', '-Wno-abi']",
         "['--compiler-options=-fPIC,-O3,-Wno-deprecated-declarations,-Wno-abi']"),
        # PPU-only extra that the CUDA build does not take
        ("    hgcc_flags.extend(['-x', 'hg'])\n", ""),
        # PPU LLVM backend flag prefix + list name (must run after the two rules above)
        ('-Xllvm', '-mllvm'),
        ('hgcc_flags', 'nvcc_flags'),
        # tmp file prefix
        ('hgcc.tmp.', 'nvcc.tmp.'),
    ],

    # --- deep_gemm/jit/__init__.py ---
    'deep_gemm/jit/__init__.py': [
        ('get_hgcc_compiler', 'get_nvcc_compiler'),
    ],

    # --- deep_gemm/jit/interleave_ffma.py ---
    'deep_gemm/jit/interleave_ffma.py': [
        # Import line: PPU_HOME variable → import from torch
        ("PPU_HOME = os.environ.get('PPU_SDK') or os.environ.get('PPU_HOME') or '/usr/local/PPU_SDK'",
         'from torch.utils.cpp_extension import CUDA_HOME'),
        # Binary path in f-string
        ('{PPU_HOME}/bin/hgobjdump', '{CUDA_HOME}/bin/cuobjdump'),
        # Function name
        ('run_hgobjdump', 'run_cuobjdump'),
    ],

    # --- csrc/utils/utils.hpp ---
    'csrc/utils/utils.hpp': [
        # In this file, the #include should be ATen/cuda/CUDAContext.h, not cuda_runtime.h
        ('#include <hggc_runtime_api.h>', '#include <ATen/cuda/CUDAContext.h>'),
    ],

    # --- tests/test_jit.py ---
    'tests/test_jit.py': [
        ('HGCC compiler:', 'NVCC compiler:'),
        ('jit.get_hgcc_compiler()', 'jit.get_nvcc_compiler()'),
    ],

    # --- csrc/jit_kernels/impls/m_grouped_int8_gemm.hpp ---
    'csrc/jit_kernels/impls/m_grouped_int8_gemm.hpp': [
        # Profiling param change: layout_info -> m_rows_tensor.data_ptr<int32_t>()
        ('expected_m, layout_info,', 'expected_m, m_rows_tensor.data_ptr<int32_t>(),'),
        # Return statement change (only the 8-space-indented occurrence)
        ('        return std::make_pair(block_m, ceil_div(n, block_n));', '        return;'),
    ],

}


# =============================================================================
# LEVEL 3b: Natural-landmark structural region replacements
# Converts genuinely divergent regions that are NOT pure token renames (e.g. the
# version query, the flags construction, the RTC include block). The PPU source
# stays completely NATURAL -- it contains NO conversion markers. Each entry is
#     (name, pattern, cuda_text)
# where `pattern` anchors on stable, self-justifying code that already exists in
# the source (a function signature, a distinctive section comment, an #if line)
# and uses non-greedy `.*?`, so a region's INTERIOR can be edited freely without
# breaking the match. `cuda_text == ''` deletes the region. Drift in a landmark
# fails loud (see apply_block_replacements + verify_required), never silent.
# =============================================================================
FILE_SPECIFIC_BLOCK_REPLACEMENTS: Dict[str, List[Tuple[str, str, str]]] = {
    'deep_gemm/jit/compiler.py': [
        # The compiler-discovery helper is a genuine rewrite (different env var, no
        # try/except, extra version assertions), so it is a structural region anchored
        # on its `def` line; the body may be edited freely.
        ('get_compiler_fn',
         r'def get_hgcc_compiler\(\) -> Tuple\[str, str\]:\n.*?(?=\n\n@functools\.lru_cache\(maxsize=None\)\ndef get_default_user_dir\(\):)',
         r"""def get_nvcc_compiler() -> Tuple[str, str]:
    paths = []
    if os.getenv('DG_NVCC_COMPILER'):
        paths.append(os.getenv('DG_NVCC_COMPILER'))
    paths.append(f'{CUDA_HOME}/bin/nvcc')

    # Try to find the first available NVCC compiler
    least_version_required = '11.6'
    version_pattern = re.compile(r'release (\d+\.\d+)')
    for path in paths:
        if os.path.exists(path):
            match = version_pattern.search(os.popen(f'{path} --version').read())
            version = match.group(1)
            assert match, f'Cannot get the version of NVCC compiler {path}'
            assert version >= least_version_required, f'NVCC {path} version {version} is lower than {least_version_required}'
            return path, version
    raise RuntimeError('Cannot find any available NVCC compiler')"""),
    ],
    'csrc/jit/compiler.hpp': [
        ('version',
         r'(?:    //[^\n]*\n)*    std::string get_hgcc_version\(\) const \{\n.*?\n    \}\n(?:    //[^\n]*\n)*',
         r"""    std::pair<int, int> get_nvcc_version() const {
        DG_HOST_ASSERT(std::filesystem::exists(nvcc_path));

        // Call the version command
        const auto& command = std::string(nvcc_path) + " --version";
        const auto& [return_code, output] = call_external_command(command);
        DG_HOST_ASSERT(return_code == 0);

        // The version should be at least 12.3, for the best performance with 12.9
        int major, minor;
        std::smatch match;
        DG_HOST_ASSERT(std::regex_search(output, match, std::regex(R"(release (\d+\.\d+))")));
        std::sscanf(match[1].str().c_str(), "%d.%d", &major, &minor);
        DG_HOST_ASSERT((major > 12 or (major == 12 and minor >= 3)) and "NVCC version should be >= 12.3");
        if (major == 12 and minor < 9)
            printf("Warning: please use at least NVCC 12.9 for the best DeepGEMM performance\n");
        return {major, minor};
    }
"""),
        # cuda-free-only trailer ("-x hg" + its comment): deleted on the CUDA side.
        ('flags_lang',
         r'\n(?:        //[^\n]*\n)*        flags \+= " -x hg ";\n(?:        //[^\n]*\n)*',
         ''),
    ],
}

# Text that MUST be present after conversion -- the single conversion check.
# If a rule stops matching, the code it was supposed to produce is simply absent,
# and that is exactly what is asserted here. Deliberately NOT a denylist of
# forbidden PPU tokens: such a list false-positives on legitimate code and stops
# the PPU source from evolving freely, which is the coupling this converter avoids.
# A rule that silently fails but leaves valid-looking code
# behind (e.g. an includes_insert list missing the CUDA-only paths) would pass it.
# This positive guard makes every token rule fail loud instead.
REQUIRED_AFTER_CONVERSION: Dict[str, List[str]] = {
    'csrc/jit/compiler.hpp': [
        '#include <ATen/cuda/CUDAContext.h>',
        '#include <cuda_runtime.h>',
        '#include <nvrtc.h>',
        'std::pair<int, int> get_nvcc_version() const {',
        'nvcc_path = sdk_home / "bin" / "nvcc";',
        'get_env<std::string>("DG_JIT_NVCC_COMPILER")',
        'const auto& [nvcc_major, nvcc_minor] = get_nvcc_version();',
        'signature = fmt::format("NVCC{}.{}", nvcc_major, nvcc_minor);',
        '-gencode=arch=compute_89,code=sm_89',
        '-gencode=arch=compute_80a,code=sm_80a',
        '-cubin --expt-relaxed-constexpr --expt-extended-lambda',
        'flags += " ";',
        '--diag-suppress=39,174,177,940',
        'if (get_env("DG_JIT_DEBUG", 0) or get_env("DG_JIT_PTXAS_VERBOSE", 0) or get_env("DG_JIT_PTXAS_CHECK", 0))',
        '-Xcompiler -O3,-Wno-deprecated-declarations,-Wno-abi',
        '-mllvm -ppu-patch-fence-ppu=false',
        'std::string nvcc_flags;',
        '#if defined(CUDA_VERSION) && CUDA_VERSION >= 13000',
        'includes_insert({cuda_home, cuda_home1, cuda_home1 + "/cuda/std"});',
        'thrust/system/cuda',
        'signature = fmt::format("NVRTC{}.{}", major, minor);',
        'and "NVRTC version should be >= 12.3"',
        'kernel.cubin',
        'nvcc_path',
    ],
    'deep_gemm/jit/compiler.py': [
        'from torch.utils.cpp_extension import CUDA_HOME',
        'def get_nvcc_compiler() -> Tuple[str, str]:',
        "paths.append(f'{CUDA_HOME}/bin/nvcc')",
        "least_version_required = '11.6'",
        "'-gencode=arch=compute_89,code=sm_89'",
        "'-gencode=arch=compute_80a,code=sm_80a'",
        "'--expt-relaxed-constexpr', '--expt-extended-lambda', '--diag-suppress=39,174,177,940',",
        "'--compiler-options=-fPIC,-O3,-Wno-deprecated-declarations,-Wno-abi'",
        "'-mllvm', '-ppu-patch-fence-ppu=false'",
        'nvcc_flags',
        'nvcc.tmp.',
    ],
    # 3.1: guards the RENAME
    'csrc/jit/device_runtime.hpp': [
        '#include <ATen/cuda/CUDAContext.h>',
        'std::shared_ptr<cudaDeviceProp> cached_prop;',
    ],
    # 3.2: the "HGGC*" labels are asserted on purpose -- they must NOT be renamed, so this
    # catches an over-eager rule.
    'csrc/utils/exception.hpp': [
        '"HGGCRTC"',
        '"HGGC driver"',
        '"HGGC runtime"',
    ],
    # setup.py: NOTE the deliberate survivors -- get_ppu_sdk_version() still shells out
    # to `hgcc --version` (works under CUDA too) and -DDG_HGGC_SUPPORT_PCH stays, so a
    # denylist on 'hgcc'/'HGGC' would have been wrong here.
    'setup.py': [
        'from torch.utils.cpp_extension import CppExtension, CUDA_HOME, CUDAExtension, BuildExtension',
        "sources = ['csrc/python_api.cu']",
        "f'{CUDA_HOME}/include',",
        "build_libraries = ['cuda', 'cudart', 'nvrtc'",
        "f'{CUDA_HOME}/lib64',",
        "f'{CUDA_HOME}/lib64/stub'",
        'extra_nvcc_args = ["-O3", "-std=c++17", "--use_fast_math"]',
        '"nvcc": extra_nvcc_args,',
        "CUDAExtension(name='deep_gemm.deep_gemm_cpp',",
    ],
    # 3.3: init() is no longer moved by the converter, but it must still be present exactly
    # once and the CUDA_HOME import must exist.
    'deep_gemm/__init__.py': [
        'from torch.version import cuda as cuda_version',
        'from packaging import version',
        'from torch.utils.cpp_extension import CUDA_HOME',
        'deep_gemm_cpp.init(',
        'CUDA_HOME         #',
    ],
}

# =============================================================================
# Helper Functions
# =============================================================================

def should_exclude(filepath: str, repo_dir: str) -> bool:
    """Check if a file should be excluded from processing."""
    rel_path = os.path.relpath(filepath, repo_dir)
    for pattern in EXCLUDE_PATTERNS:
        if pattern.endswith('/'):
            if rel_path.startswith(pattern) or f'/{pattern}' in f'/{rel_path}':
                return True
        elif pattern.startswith('*.'):
            ext = pattern[1:]
            if rel_path.endswith(ext):
                return True
        elif pattern in rel_path:
            return True
    return False


def get_target_files(repo_dir: str) -> List[str]:
    """Get list of files to process."""
    files = []
    for root, dirs, filenames in os.walk(repo_dir):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if not should_exclude(os.path.join(root, d), repo_dir)]
        for fname in filenames:
            filepath = os.path.join(root, fname)
            if should_exclude(filepath, repo_dir):
                continue
            ext = os.path.splitext(fname)[1]
            if ext in TARGET_EXTENSIONS:
                files.append(filepath)
    return sorted(files)


def apply_replacements(content: str, replacements: List[Tuple[str, str]]) -> str:
    """Apply a list of (old, new) text replacements to content."""
    for old, new in replacements:
        if old and old != new:
            # Idempotency guard: for insertion-type rules where `old` is a substring
            # of `new` (e.g. inserting a comment/macro block by re-emitting the anchor
            # plus extra content), skip if the result is already present. This prevents
            # duplicate insertions when the script is run more than once.
            if old in new and new in content:
                continue
            content = content.replace(old, new)
    return content


def apply_regex_replacements(content: str, regex_list: List[Tuple[str, str, str]], filepath: str) -> str:
    """Apply regex-based replacements."""
    for pattern, replacement, file_glob in regex_list:
        if file_glob and not filepath.endswith(file_glob):
            continue
        content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    return content


def apply_block_replacements(content: str, blocks: List[Tuple[str, str, str]], rel_path: str) -> str:
    """Replace structural regions matched on NATURAL code landmarks.

    Each entry is (name, pattern, cuda_text). `pattern` anchors on stable natural
    code in the PPU source (a function signature, a distinctive section comment,
    an #if line) and uses non-greedy `.*?`, so editing a region's INTERIOR never
    breaks the match -- only the landmark lines are load-bearing. The whole
    matched span is replaced by `cuda_text` ('' deletes the region). A landmark
    that no longer matches is a hard error (unless already converted), so drift
    fails loud instead of silently emitting a half-converted file.
    """
    for name, pattern, cuda_text in blocks:
        rx = re.compile(pattern, re.DOTALL)
        new_content, n = rx.subn(lambda m: cuda_text, content)
        if n == 0:
            # Landmark not found: tolerate only if clearly already converted.
            if cuda_text == '':
                continue
            if cuda_text.strip() and cuda_text in content:
                continue
            raise RuntimeError(
                "[cuda_compat] Structural region '%s' not found in %s. Its natural landmark "
                "lines were altered or removed; refusing to emit a half-converted file. "
                "Update its pattern in FILE_SPECIFIC_BLOCK_REPLACEMENTS or restore the "
                "landmark." % (name, rel_path))
        elif n > 1:
            raise RuntimeError(
                "[cuda_compat] Structural region '%s' matched %d times in %s (ambiguous "
                "landmark). Tighten its pattern in FILE_SPECIFIC_BLOCK_REPLACEMENTS." % (
                    name, n, rel_path))
        content = new_content
    return content


def verify_required(content: str, rel_path: str) -> None:
    """Fail loud if expected CUDA text is missing after conversion (silent under-conversion)."""
    required = REQUIRED_AFTER_CONVERSION.get(rel_path)
    if not required:
        return
    missing = [s for s in required if s not in content]
    if missing:
        raise RuntimeError(
            "[cuda_compat] Converted %s is missing expected CUDA text:\n%s\n"
            "  A conversion rule silently failed to match -- the result may still be\n"
            "  syntactically valid, it just lost the ported behaviour. Check the "
            "corresponding rule in FILE_SPECIFIC_REPLACEMENTS / "
            "FILE_SPECIFIC_BLOCK_REPLACEMENTS." % (
                rel_path, "\n".join("    missing: %r" % s for s in missing[:20])))


def verify_python_syntax(content: str, rel_path: str, original: str = None) -> None:
    """Fail loud if conversion BROKE a .py file's syntax.

    Substring replacements can silently corrupt indentation (e.g. a rule whose match
    text carries fewer leading spaces than the line it hits leaves stray whitespace
    behind). The token/required guards work on substrings and cannot see that, so parse
    the result instead.

    This is deliberately a DIFFERENTIAL check. compile() runs under whatever interpreter
    executes this script, which may be older than the Python the sources target -- e.g.
    Python 3.6 rejects the valid 3.8+ self-documenting f-string `f'{x=}'`. Judging the
    converted file on its own therefore produces false alarms. Instead the original is
    parsed too, and a failure is only reported when the original parsed and the
    converted one does not, which pins the blame on conversion.
    """
    if not rel_path.endswith('.py'):
        return
    try:
        compile(content, rel_path, 'exec')
        return                                    # converted file parses: nothing to do
    except SyntaxError as e:
        converted_err = e

    if original is not None:
        try:
            compile(original, rel_path, 'exec')
        except SyntaxError:
            # The source does not parse under this interpreter either, so the converter
            # is not at fault (most likely a language feature newer than this Python).
            return

    raise RuntimeError(
        "[cuda_compat] Conversion broke the syntax of %s: %s (line %s).\n"
        "  The file parsed before conversion but not after, so a replacement rule\n"
        "  corrupted it -- most often a rule whose match text has a different\n"
        "  indentation than the line it matched." % (
            rel_path, converted_err.msg, converted_err.lineno))


# =============================================================================
# Main Processing
# =============================================================================

def process_file(filepath: str, repo_dir: str, dry_run: bool = False, verbose: bool = False, reference_dir: str = None) -> int:
    """Process a single file, return number of replacements made."""
    rel_path = os.path.relpath(filepath, repo_dir)

    # Check for full file copy from reference directory
    # When reference_dir is available, ALL files are copied from it for exact alignment
    if reference_dir:
        ref_file = os.path.join(reference_dir, rel_path)
        if os.path.exists(ref_file):
            try:
                with open(ref_file, 'r', encoding='utf-8') as f:
                    ref_content = f.read()
                with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                    original = f.read()
            except (IOError, OSError) as e:
                print(f"  [WARN] Cannot read files for {rel_path}: {e}")
                return 0
            if original == ref_content:
                return 0
            if dry_run:
                print(f"  [DRY-RUN] Would copy from reference: {rel_path}")
            else:
                try:
                    # Ensure writable
                    if not os.access(filepath, os.W_OK):
                        os.chmod(filepath, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(ref_content)
                    if verbose:
                        print(f"  [COPIED] {rel_path} (from reference directory)")
                except PermissionError:
                    print(f"  [WARN] Permission denied, skipping: {rel_path}")
                    return 0
            return 1
        # File not in reference dir - fall through to text replacement

    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            original = f.read()
    except (IOError, OSError) as e:
        print(f"  [WARN] Cannot read {filepath}: {e}")
        return 0

    content = original

    # 0. Apply anchor-delimited structural block replacements (robust to interior edits)
    if rel_path in FILE_SPECIFIC_BLOCK_REPLACEMENTS:
        content = apply_block_replacements(content, FILE_SPECIFIC_BLOCK_REPLACEMENTS[rel_path], rel_path)

    # 1. Apply file-specific replacements first (highest priority)
    if rel_path in FILE_SPECIFIC_REPLACEMENTS:
        content = apply_replacements(content, FILE_SPECIFIC_REPLACEMENTS[rel_path])

    # 2. Apply include replacements
    content = apply_replacements(content, INCLUDE_REPLACEMENTS)

    # 2b. Apply acBLAS -> cuBLAS replacements
    content = apply_replacements(content, BLAS_REPLACEMENTS)

    # 3. Apply enum/constant replacements (before shorter patterns)
    content = apply_replacements(content, ENUM_REPLACEMENTS)

    # 4. Apply type replacements
    content = apply_replacements(content, TYPE_REPLACEMENTS)

    # 5. Apply function replacements
    content = apply_replacements(content, FUNC_REPLACEMENTS)

    # 6. Apply macro replacements
    content = apply_replacements(content, MACRO_REPLACEMENTS)

    # 7. Apply Python-specific replacements (only for .py files)
    if filepath.endswith('.py'):
        content = apply_replacements(content, PYTHON_REPLACEMENTS)

    # 8. Apply regex replacements
    content = apply_regex_replacements(content, REGEX_REPLACEMENTS, filepath)

    # 9. Robustness guard: fail loud if PPU/HGGC tokens survived in guarded files
    verify_required(content, rel_path)
    verify_python_syntax(content, rel_path, original)

    # Check if anything changed
    if content == original:
        return 0

    # Count approximate number of replacements
    # (simple heuristic: count differing lines)
    orig_lines = original.splitlines()
    new_lines = content.splitlines()
    changes = sum(1 for a, b in zip(orig_lines, new_lines) if a != b)
    changes += abs(len(orig_lines) - len(new_lines))

    if dry_run:
        if verbose or changes > 0:
            print(f"  [DRY-RUN] Would modify: {rel_path} ({changes} line changes)")
    else:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        if verbose:
            print(f"  [MODIFIED] {rel_path} ({changes} line changes)")

    return changes

def handle_file_renames(repo_dir: str, dry_run: bool = False, verbose: bool = False) -> int:
    """Handle file rename operations."""
    changes = 0
    for old_rel, new_rel in FILE_RENAMES:
        old_path = os.path.join(repo_dir, old_rel)
        new_path = os.path.join(repo_dir, new_rel)
        if os.path.exists(old_path) and not os.path.exists(new_path):
            # Only .cpp exists -> rename to .cu
            if dry_run:
                print(f"  [DRY-RUN] Would rename: {old_rel} -> {new_rel}")
            else:
                os.rename(old_path, new_path)
                print(f"  [RENAMED] {old_rel} -> {new_rel}")
            changes += 1
        elif os.path.exists(old_path) and os.path.exists(new_path):
            # Both exist -> remove .cpp (the .cu is the original)
            if dry_run:
                print(f"  [DRY-RUN] Would remove: {old_rel} (keeping {new_rel})")
            else:
                os.remove(old_path)
                print(f"  [REMOVED] {old_rel} (keeping {new_rel})")
            changes += 1
        elif os.path.exists(new_path):
            if verbose:
                print(f"  [SKIP] {new_rel} already exists, {old_rel} not found")
    return changes


def restore_cmakelists(repo_dir: str, dry_run: bool = False) -> int:
    """Restore CMakeLists.txt that was deleted in ppu-original version."""
    cmake_path = os.path.join(repo_dir, 'CMakeLists.txt')
    if os.path.exists(cmake_path):
        print("  [SKIP] CMakeLists.txt already exists")
        return 0

    cmake_content = """\
# NOTES: current just for CMake-based IDE (e.g. CLion) indexing, the real compilation is done via JIT
# TODO: add CUDA utils' library via CMake
cmake_minimum_required(VERSION 3.10)
project(deep_gemm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(CUDAToolkit REQUIRED)
find_package(pybind11 REQUIRED)

file(WRITE ${CMAKE_BINARY_DIR}/test_cuda.cu "extern \\"C\\" __global__ void testKernel() { }")
execute_process(
        COMMAND ${CUDA_NVCC_EXECUTABLE} ${CMAKE_CUDA_FLAGS} -gencode arch=compute_90a,code=sm_90a -o ${CMAKE_BINARY_DIR}/test_cuda.o -c ${CMAKE_BINARY_DIR}/test_cuda.cu
        RESULT_VARIABLE NVCC_RESULT
        OUTPUT_VARIABLE NVCC_OUTPUT
        ERROR_VARIABLE NVCC_ERROR_OUTPUT
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

if (NVCC_RESULT EQUAL "0")
    set(NVCC_SUPPORTS_SM90 TRUE)
    message(STATUS "NVCC supports SM90")
else()
    message(STATUS "NVCC does not support SM90")
endif()

if (NVCC_SUPPORTS_SM90)
    set(TORCH_CUDA_ARCH_LIST "8.6" CACHE STRING "Add arch tag 90a to NVCC" FORCE)
    list(APPEND CUDA_NVCC_FLAGS "-gencode;arch=compute_90a,code=sm_90a")
endif()
find_package(Torch REQUIRED)

include_directories(deep_gemm/include third-party/cutlass/include third-party/cutlass/tools/util/include)
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include ${TORCH_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
link_directories(${TORCH_INSTALL_PREFIX}/lib ${CUDA_TOOLKIT_ROOT_DIR}/lib)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -fPIC -DNDEBUG")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -std=c++17 -DNDEBUG")

cuda_add_library(example_gemm STATIC indexing/main.cu)
"""
    if dry_run:
        print("  [DRY-RUN] Would restore CMakeLists.txt")
    else:
        with open(cmake_path, 'w') as f:
            f.write(cmake_content)
        print("  [RESTORED] CMakeLists.txt")
    return 1


def main():
    parser = argparse.ArgumentParser(
        description='Revert PPU/HGGC ppu-original code back to standard CUDA')
    parser.add_argument('repo_dir', nargs='?', default=DEFAULT_REPO_DIR,
                        help=f'Path to DeepGemm repository (default: {DEFAULT_REPO_DIR})')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without making changes')
    parser.add_argument('--reference-dir', default=None,
                        help=f'Path to original DeepGemm for full-file copy (default: auto-detect)')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed output')
    args = parser.parse_args()

    repo_dir = os.path.abspath(args.repo_dir)
    if not os.path.isdir(repo_dir):
        print(f"ERROR: Repository directory not found: {repo_dir}")
        sys.exit(1)

    # Resolve reference directory (only if explicitly provided via CLI)
    reference_dir = None
    if args.reference_dir:
        reference_dir = os.path.abspath(args.reference_dir)
        if not os.path.isdir(reference_dir):
            print(f"  [WARN] Reference directory not found: {reference_dir}")
            reference_dir = None

    if reference_dir:
        # Validate: must be the ORIGINAL CUDA version, not another ppu-original copy
        ref_setup = os.path.join(reference_dir, 'setup.py')
        if os.path.isfile(ref_setup):
            try:
                with open(ref_setup, 'r', encoding='utf-8', errors='replace') as f:
                    setup_content = f.read()
                if 'CppExtension(' in setup_content and 'CUDAExtension(' not in setup_content:
                    print(f"  [WARN] Reference dir appears to be ppu-original version, disabling.")
                    reference_dir = None
            except (IOError, OSError):
                pass
        try:
            if reference_dir and os.path.samefile(reference_dir, repo_dir):
                print(f"  [WARN] Reference dir is same as repo dir, disabling.")
                reference_dir = None
        except OSError:
            pass
    if reference_dir:
        print(f"  Reference dir: {reference_dir}")
    else:
        print(f"  Mode: Self-contained text replacement (no reference directory)")

    print(f"{'='*70}")
    print(f"  Revert PPU-original -> CUDA Compatible")
    print(f"  Repository: {repo_dir}")
    print(f"  Mode: {'DRY-RUN' if args.dry_run else 'APPLY'}")
    print(f"{'='*70}")

    total_changes = 0

    # --- Phase 1: File operations ---
    print("\n[Phase 1] File operations (deletions, renames, restores)...")
    total_changes += handle_file_renames(repo_dir, args.dry_run, args.verbose)
    total_changes += restore_cmakelists(repo_dir, args.dry_run)

    # --- Phase 2: Text replacements ---
    print("\n[Phase 2] Text replacements...")
    files = get_target_files(repo_dir)
    print(f"  Found {len(files)} target files to process")

    files_modified = 0
    for filepath in files:
        changes = process_file(filepath, repo_dir, args.dry_run, args.verbose, reference_dir)
        if changes > 0:
            files_modified += 1
            total_changes += changes

    # --- Summary ---
    print(f"\n{'='*70}")
    print(f"  Summary:")
    print(f"    Files scanned:  {len(files)}")
    print(f"    Files modified: {files_modified}")
    print(f"    Total changes:  {total_changes}")
    if args.dry_run:
        print(f"    (DRY-RUN mode - no actual changes made)")
    print(f"{'='*70}")

    return 0


if __name__ == '__main__':
    sys.exit(main())

