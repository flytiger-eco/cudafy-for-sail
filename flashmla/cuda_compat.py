#!/usr/bin/env python3
"""
FlashMLA PPU-original → CUDA-Compatible Transformation Script.

Refactored with lookup-table approach for maintainability.
Converts a FlashMLA ppu-original source tree into a CUDA-compatible one.

Usage:
    python cuda_compat.py [TARGET_DIR] [--dry-run] [--verbose]
"""

import argparse
import os
import re
import sys
import time
from typing import List, Tuple, Dict

# =============================================================================
# Lookup tables — organized by category.
# Within each category, longer/more-specific patterns MUST come first
# to avoid partial matches on shorter substrings.
# =============================================================================

INCLUDE_REPLACEMENTS: List[Tuple[str, str]] = [
    ('<hggc_runtime.h>', '<cuda_runtime.h>'),
    ('<hggc_fp16.h>', '<cuda_fp16.h>'),
    ('<hggc_bf16.h>', '<cuda_bf16.h>'),
    ('<hggc_pipeline.h>', '<cuda_pipeline.h>'),
    ('<hggc_awbarrier.h>', '<cuda_awbarrier.h>'),
    ('<hggc_ad.h>', '"cuda_ad.h"'),
    ('"hggc_runtime.h"', '"cuda_runtime.h"'),
    ('<hgtx3/hgToolsExt.h>', '<nvtx3/nvToolsExt.h>'),
]

ENUM_REPLACEMENTS: List[Tuple[str, str]] = [
    # Longer compound names first (may embed type/function prefixes)
    ('hggcLaunchAttributeProgrammaticStreamSerialization', 'cudaLaunchAttributeProgrammaticStreamSerialization'),
    ('hggcFuncAttributeMaxDynamicSharedMemorySize', 'cudaFuncAttributeMaxDynamicSharedMemorySize'),
    ('hggcOccupancyMaxActiveBlocksPerMultiprocessor', 'cudaOccupancyMaxActiveBlocksPerMultiprocessor'),
    ('hggcOccupancyDisableCachingOverride', 'cudaOccupancyDisableCachingOverride'),
    ('hggcDevAttrMaxSharedMemoryPerMultiprocessor', 'cudaDevAttrMaxSharedMemoryPerMultiprocessor'),
    ('hggcDevAttrMaxSharedMemoryPerBlockOptin', 'cudaDevAttrMaxSharedMemoryPerBlockOptin'),
    ('hggcDevAttrMultiProcessorCount', 'cudaDevAttrMultiProcessorCount'),
    ('hggcDevAttrComputeCapabilityMajor', 'cudaDevAttrComputeCapabilityMajor'),
    ('hggcDevAttrComputeCapabilityMinor', 'cudaDevAttrComputeCapabilityMinor'),
    ('hggcStreamCaptureModeRelaxed', 'cudaStreamCaptureModeRelaxed'),
    ('hggcStreamCaptureModeNone', 'cudaStreamCaptureStatusNone'),
    ('hggcStreamCaptureStatusNone', 'cudaStreamCaptureStatusNone'),
    ('hggcStreamNonBlocking', 'cudaStreamNonBlocking'),
    ('hggcStreamDefault', 'cudaStreamDefault'),
    ('hggcMemcpyDeviceToHost', 'cudaMemcpyDeviceToHost'),
    ('hggcMemcpyDeviceToDevice', 'cudaMemcpyDeviceToDevice'),
    ('hggcMemcpyHostToDevice', 'cudaMemcpyHostToDevice'),
    ('hggcMemcpyHostToHost', 'cudaMemcpyHostToHost'),
    ('hggcErrorUnknown', 'cudaErrorUnknown'),
    ('HGGC_SUCCESS', 'CUDA_SUCCESS'),
    ('HGAD_LAUNCH_ATTRIBUTE_IGNORE', 'CUAD_LAUNCH_ATTRIBUTE_IGNORE'),
    # Short patterns last
    ('hggcSuccess', 'cudaSuccess'),
    ('hggcError', 'cudaError'),
]

TYPE_REPLACEMENTS: List[Tuple[str, str]] = [
    ('hggcLaunchConfig_t', 'cudaLaunchConfig_t'),
    ('hggcLaunchAttribute', 'cudaLaunchAttribute'),
    ('hggcStream_t', 'cudaStream_t'),
    ('hggcFunction_t', 'cudaFunction_t'),
    ('hggcDataType_t', 'cudaDataType_t'),
    ('hggcEvent_t', 'cudaEvent_t'),
    ('hggcFuncAttributes', 'cudaFuncAttributes'),
    ('hggcFuncAttribute', 'cudaFuncAttribute'),
    ('hggcMemcpyKind', 'cudaMemcpyKind'),
    ('hggcStreamCaptureStatus', 'cudaStreamCaptureStatus'),
    ('hggcStreamCaptureMode', 'cudaStreamCaptureMode'),
    ('HGlaunchAttributeAD', 'CUlaunchAttributeAD'),
    ('HGlaunchConfigAD', 'CUlaunchConfigAD'),
    ('HGfunction', 'CUfunction'),
    ('HGresult', 'CUresult'),
    ('__hg_fp8', '__nv_fp8'),
]

# bfloat16 type names require word-boundary matching so that e.g.
# __ppu_bfloat162 does not also match inside __ppu_bfloat162_raw.
WORD_BOUNDARY_REPLACEMENTS: List[Tuple[str, str]] = [
    ('__ppu_bfloat162', '__nv_bfloat162'),
    ('__ppu_bfloat16_raw', '__nv_bfloat16_raw'),
    ('__ppu_bfloat16', '__nv_bfloat16'),
]

FUNC_REPLACEMENTS: List[Tuple[str, str]] = [
    ('hggcOccupancyMaxPotentialBlockSize', 'cudaOccupancyMaxPotentialBlockSize'),
    ('hggcTriggerProgrammaticLaunchCompletion', 'cudaTriggerProgrammaticLaunchCompletion'),
    ('hggcGridDependencySynchronize', 'cudaGridDependencySynchronize'),
    ('hggcDeviceGetAttribute', 'cudaDeviceGetAttribute'),
    ('hggcGetDeviceProperties', 'cudaGetDeviceProperties'),
    ('hggcGetFuncBySymbol', 'cudaGetFuncBySymbol'),
    ('hggcFuncSetAttribute', 'cudaFuncSetAttribute'),
    ('hggcFuncGetAttributes', 'cudaFuncGetAttributes'),
    ('hggcGetErrorString', 'cudaGetErrorString'),
    ('hggcGetLastError', 'cudaGetLastError'),
    ('hggcGetErrorName', 'cudaGetErrorName'),
    ('hggcPeekAtLastError', 'cudaPeekAtLastError'),
    ('hggcGetDevice', 'cudaGetDevice'),
    ('hggcLaunchKernelExAD', 'cuLaunchKernelExAD'),
    ('hggcLaunchKernelExC', 'cudaLaunchKernelEx'),
    ('hggcLaunchKernelEx', 'cudaLaunchKernelEx'),
    ('hggcLaunchKernel', 'cudaLaunchKernel'),
    ('hggcThreadExchangeStreamCaptureMode', 'cudaThreadExchangeStreamCaptureMode'),
    ('hggcStreamCreateWithFlags', 'cudaStreamCreateWithFlags'),
    ('hggcStreamIsCapturing', 'cudaStreamIsCapturing'),
    ('hggcStreamSynchronize', 'cudaStreamSynchronize'),
    ('hggcStreamDestroy', 'cudaStreamDestroy'),
    ('hggcMemcpyAsync', 'cudaMemcpyAsync'),
    ('hggcMemsetAsync', 'cudaMemsetAsync'),
    ('hggcDeviceSynchronize', 'cudaDeviceSynchronize'),
    ('hggcSetDevice', 'cudaSetDevice'),
    ('hggcMemGetInfo', 'cudaMemGetInfo'),
    ('hggcMalloc', 'cudaMalloc'),
    ('hggcFree', 'cudaFree'),
    ('hggcMemcpy', 'cudaMemcpy'),
    ('hggcEventElapsedTime', 'cudaEventElapsedTime'),
    ('hggcEventSynchronize', 'cudaEventSynchronize'),
    ('hggcEventCreate', 'cudaEventCreate'),
    ('hggcEventDestroy', 'cudaEventDestroy'),
    ('hggcEventRecord', 'cudaEventRecord'),
    ('hgLaunchKernelExAD', 'cuLaunchKernelExAD'),
    ('hgGetErrorName', 'cuGetErrorName'),
    ('hgGetErrorString', 'cuGetErrorString'),
]

MACRO_REPLACEMENTS: List[Tuple[str, str]] = [
    ('__HGGC_ARCH__', '__CUDA_ARCH__'),
]

NVTX_REPLACEMENTS: List[Tuple[str, str]] = [
    ('hgtxEventAttributes_t', 'nvtxEventAttributes_t'),
    ('hgtxDomainHandle_t', 'nvtxDomainHandle_t'),
    ('hgtxDomainCreateA', 'nvtxDomainCreateA'),
    ('hgtxDomainDestroy', 'nvtxDomainDestroy'),
    ('hgtxDomainRangePushEx', 'nvtxDomainRangePushEx'),
    ('hgtxDomainRangePop', 'nvtxDomainRangePop'),
    ('HGTX_MESSAGE_TYPE_ASCII', 'NVTX_MESSAGE_TYPE_ASCII'),
    ('HGTX_VERSION', 'NVTX_VERSION'),
    ('use_hgtx_', 'use_nvtx_'),
]

# Processing order: include → enum → type → func → macro → nvtx
ALL_CATEGORY_REPLACEMENTS = [
    INCLUDE_REPLACEMENTS,
    ENUM_REPLACEMENTS,
    TYPE_REPLACEMENTS,
    FUNC_REPLACEMENTS,
    MACRO_REPLACEMENTS,
    NVTX_REPLACEMENTS,
]

# Arch value reversals: applied only on lines containing __CUDA_ARCH__
ARCH_VALUE_REPLACEMENTS = [
    ('>= 100', '>= 800'),
    ('== 100', '== 800'),
    ('== 150', '== 890'),
]

# =============================================================================
# Regex-based replacements: (pattern, replacement, file_glob)
# file_glob is matched against filepath via endswith()
# =============================================================================

REGEX_REPLACEMENTS: List[Tuple[str, str, str]] = [
    # combine.cuh: strip (const void*) and (void**)& casts from cudaLaunchKernelEx args
    (r'cudaLaunchKernelEx\(([^,]+),\s*\(const void\*\)([^,]+),\s*\(void\*\*\)&([^)]+)\)',
     r'cudaLaunchKernelEx(\1, \2, \3)', 'combine.cuh'),

    # C10_CUDA_CHECK wrapping for cudaFuncSetAttribute (single-line)
    (r'^([ \t]+)cudaFuncSetAttribute\(([^)]+)\);',
     r'\1C10_CUDA_CHECK(cudaFuncSetAttribute(\2));', 'splitkv_mla.cuh'),
    (r'^([ \t]+)cudaFuncSetAttribute\(([^)]+)\);',
     r'\1C10_CUDA_CHECK(cudaFuncSetAttribute(\2));', 'splitkv_mla_kernel.cuh'),
    (r'^([ \t]+)cudaFuncSetAttribute\(([^)]+)\);',
     r'\1C10_CUDA_CHECK(cudaFuncSetAttribute(\2));', 'sparse_prefill_wg.cuh'),

    # SM detection for splitkv_mla_kernel
    (r'            int sm_count = get_num_sm\(get_current_device\(\)\);\s*\n'
     r'            if \(sm_count == 64\) sm_count = 20;',
     r'            auto dprops = at::cuda::getCurrentDeviceProperties();\n'
     r'\n'
     r'            int sm_count = dprops->multiProcessorCount;\n'
     r'            if (std::string(dprops->name).find("810E") != std::string::npos) {\n'
     r'                sm_count = 20;\n'
     r'            }',
     'splitkv_mla_kernel.cuh'),

    # SM detection for sparse_prefill_wg
    (r'        int sm_count = get_num_sm\(get_current_device\(\)\);\s*\n'
     r'            if \(sm_count == 64\) sm_count = 20;',
     r'        auto dprops = at::cuda::getCurrentDeviceProperties();\n'
     r'\n'
     r'            int sm_count = dprops->multiProcessorCount;\n'
     r'            if (std::string(dprops->name).find("810E") != std::string::npos) {\n'
     r'                sm_count = 20;\n'
     r'            }',
     'sparse_prefill_wg.cuh'),

    # SM detection for splitkv_mla.h
    (r'    \{\n'
     r'        auto \[cap_major, cap_minor\] = get_compute_capability\(get_current_device\(\)\);\s*\n'
     r'        if \(cap_major < 8 \|\| \(cap_major == 8 && cap_minor < 9\)\)\s*\n'
     r'            warp_interleave = false;\s*\n'
     r'    \}',
     r'    auto dprops = at::cuda::getCurrentDeviceProperties();\n'
     r'    if (std::string(dprops->name).find("610") != std::string::npos)\n'
     r'        warp_interleave = false;',
     'splitkv_mla.h'),

    # SM detection for utils.h
    (r'            int sm_count = get_num_sm\(get_current_device\(\)\);\s*\n'
     r'            if \(sm_count == 64\) sm_count = 20;',
     r'            auto dprops = at::cuda::getCurrentDeviceProperties();\n'
     r'            int sm_count = dprops->multiProcessorCount == 64 ? 20 : dprops->multiProcessorCount;',
     'utils.h'),

    # kernel_traits.h: remove extra SmemCopyAtomQ/K fallback
    (r'(using SmemCopyAtomTransposed = Copy_Atom<DefaultCopy, elem_type>;\n)'
     r'    using SmemCopyAtomQ = SmemCopyAtom;\n'
     r'    using SmemCopyAtomK = SmemCopyAtom;\n',
     r'\1', 'kernel_traits.h'),
]

# =============================================================================
# File-specific replacements: {relative_path: [(old, new), ...]}
# Applied BEFORE category tables so they can override general patterns.
# =============================================================================

FILE_SPECIFIC_REPLACEMENTS: Dict[str, List[Tuple[str, str]]] = {

    # --- kerutils host/host.h ---
    'csrc/kerutils/include/kerutils/host/host.h': [
        (
            '#include <hggc_runtime.h>',
            '#include <cuda_runtime_api.h>\n#include <ATen/cuda/CUDAContext.h>'
        ),
        (
            'static inline bool is_sm89_or_newer() {\n'
            '    int device = 0;\n'
            '    hggcGetDevice(&device);\n'
            '    int major = 0, minor = 0;\n'
            '    hggcDeviceGetAttribute(&major, hggcDevAttrComputeCapabilityMajor, device);\n'
            '    hggcDeviceGetAttribute(&minor, hggcDevAttrComputeCapabilityMinor, device);\n'
            '    return (major > 8) || (major == 8 && minor >= 9);\n'
            '}',
            'static inline bool is_sm89_or_newer() {\n'
            '    auto dprops = at::cuda::getCurrentDeviceProperties();\n'
            '    return (dprops->major > 8) || (dprops->major == 8 && dprops->minor >= 9);\n'
            '}'
        ),
    ],

    # --- kerutils common/common.h ---
    'csrc/kerutils/include/kerutils/common/common.h': [
        (
            '#include <hggc_runtime.h>',
            '#if !defined(__CUDACC_RTC__)\n#include "cuda_runtime.h"\n#endif'
        ),
        (
            'fprintf(stderr, "HGGC error (%s:%d): %s\\n", __FILE__, __LINE__,',
            'fprintf(stderr, "CUDA error (%s:%d): %s\\n", __FILE__, __LINE__,'
        ),
    ],

    # --- kerutils host/hardware_info.h ---
    'csrc/kerutils/include/kerutils/host/hardware_info.h': [
        ('#include <cstdio>\n#include <cstdlib>\n', ''),
        (
            '#include <hggc_runtime.h>',
            '#if !defined(__CUDACC_RTC__)\n#include "cuda_runtime.h"\n#endif'
        ),
        # Drop the duplicate CHECK_CUDA definition, common.h already provides it
        (
            '#define CHECK_CUDA(call)                                                       \\\n'
            '  do {                                                                         \\\n'
            '    hggcError_t status_ = call;                                                \\\n'
            '    if (status_ != hggcSuccess) {                                              \\\n'
            '      fprintf(stderr, "HGGC error (%s:%d): %s\\n", __FILE__, __LINE__,          \\\n'
            '              hggcGetErrorString(status_));                                    \\\n'
            '      exit(1);                                                                 \\\n'
            '    }                                                                          \\\n'
            '  } while (0)\n'
            '\n',
            ''
        ),
    ],

    # --- api/common.h ---
    'csrc/api/common.h': [
        # Remove hggcStream_t forward declaration
        (
            '#include <c10/cuda/CUDAStream.h>\n'
            '// Forward-declare hggcStream_t so function signatures match between .cu and .cpp\n'
            'typedef struct HGstream_st* hggcStream_t;',
            '#include <c10/cuda/CUDAStream.h>'
        ),
    ],

    # --- csrc/params.h (fp16/bf16/runtime headers handled by INCLUDE table) ---
    'csrc/params.h': [
        (
            '#include <hggc_runtime.h>',
            '#include <hggc_runtime.h>\n#include <ATen/cuda/CUDAContext.h>'
        ),
    ],

    # --- csrc/utils.h (fp16/bf16 headers handled by INCLUDE table) ---
    'csrc/utils.h': [
        (
            '        printf("HG driver error: %s: %s\\n",                \\\n'
            '               (_name ? _name : "?"), (_str ? _str : "?")); \\\n',
            '        TORCH_CHECK(false, "CUDA driver error ",            \\\n'
            '        (_name ? _name : "?"), ": ", (_str ? _str : "?"));  \\\n'
        ),
    ],

    # --- __HGGCCC__ guards around cuda_ad.h are applied post-conversion
    #     (see apply_hggccc_guards) because the input uses <hggc_ad.h>. ---
}

# Files whose cuda_ad.h include must be wrapped in #ifdef __HGGCCC__ for nvcc.
HGGCCC_GUARD_FILES = {
    'csrc/ppu/decode/dense/splitkv_mla_kernel.cuh',
    'csrc/ppu/decode/sparse/splitkv_mla.cuh',
    'csrc/ppuxx/decode/combine/combine.cuh',
}

# =============================================================================
# setup.py replacements
# =============================================================================

SETUP_PY_REPLACEMENTS: List[Tuple[str, str]] = [
    # --- ppu arch flags -> gencode ---
    ('cc_flag.append("-arch=ppu_10")\n'
     'cc_flag.append("-arch=ppu_15")\n',
     'cc_flag.append("-gencode")\n'
     'cc_flag.append("arch=compute_80,code=sm_80")\n'),

    # --- cuda driver library ---
    ('        sources=get_sources(),\n'
     '        extra_compile_args={\n',
     '        sources=get_sources(),\n'
     "        libraries=['cuda'],\n"
     '        extra_compile_args={\n'),

    # --- ptxas resource report + register usage level ---
    ('                    "--use_fast_math",\n'
     '                    "-mllvm",\n',
     '                    "--use_fast_math",\n'
     '                    "--ptxas-options=-v,--register-usage-level=10",\n'
     '                    "-mllvm",\n'),
]


def rewrite_setup_py(content: str) -> str:
    """Convert setup.py from the ppu-original form to the cuda-compatible form."""
    content = content.replace('hgcc', 'nvcc')
    return apply_replacements(content, SETUP_PY_REPLACEMENTS)


# =============================================================================
# Files to skip entirely (no conversion needed)
# =============================================================================

EXCLUDE_FILES = [
    'csrc/ppu/acc_vreg_fraga.h',  # uses __HGGC_ARCH__ in both modes
]

# =============================================================================
# Helper functions
# =============================================================================

def apply_replacements(content: str, replacements: List[Tuple[str, str]]) -> str:
    """Apply a list of (old, new) text replacements to content.

    Idempotency guard: if old is a substring of new AND new is already in content,
    skip the replacement (prevents double-insertion of guards/wrappers).
    """
    for old, new in replacements:
        if old and old != new:
            if old in new and new in content:
                continue
            content = content.replace(old, new)
    return content


def apply_arch_value_replacements(content: str) -> str:
    """Replace __CUDA_ARCH__ arch values: hggc arch → cuda arch.
    Also handles the LDSM/SmemCopyAtom special case (>= 800 → >= 750).
    """
    lines = content.split('\n')
    new_lines = []
    for i, line in enumerate(lines):
        if '__CUDA_ARCH__' in line:
            for old, new in ARCH_VALUE_REPLACEMENTS:
                line = line.replace(old, new)
            next_line = lines[i + 1] if i + 1 < len(lines) else ''
            if 'LDSM' in next_line or 'SmemCopyAtom' in next_line:
                line = line.replace('>= 800', '>= 750')
        new_lines.append(line)
    return '\n'.join(new_lines)


def apply_regex_replacements(content: str, regex_list: List[Tuple[str, str, str]],
                             filepath: str) -> str:
    """Apply regex-based replacements. Each entry is (pattern, replacement, file_glob).
    file_glob is matched against filepath via endswith(); None means apply to all files.
    """
    for pattern, replacement, file_glob in regex_list:
        if file_glob and not filepath.endswith(file_glob):
            continue
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)
    return content


def apply_word_boundary_replacements(content: str,
                                     replacements: List[Tuple[str, str]]) -> str:
    """Apply (old, new) replacements using word-boundary regex.

    Required for type names such as __ppu_bfloat162 so they do not also match
    inside __ppu_bfloat162_raw (which str.replace would do as a prefix match).
    """
    for old, new in replacements:
        if old and old != new:
            content = re.sub(r'\b' + re.escape(old) + r'\b', new, content)
    return content


def apply_hggccc_guards(content: str, rel_path: str) -> str:
    """Wrap cuda_ad.h includes in #ifdef __HGGCCC__ guards for nvcc (nvcc does
    not define __HGGCCC__). Must run after <hggc_ad.h> -> "cuda_ad.h" conversion.

    Files that already contain an __HGGCCC__ guard elsewhere are left untouched,
    matching the reference behaviour (combine.cuh / splitkv_mla.cuh already
    guard other blocks, so their cuda_ad.h include stays unguarded)."""
    if rel_path not in HGGCCC_GUARD_FILES:
        return content
    if '#ifdef __HGGCCC__' in content:
        return content
    content = content.replace(
        '#include "cuda_ad.h"\n#include "utils.h"',
        '#ifdef __HGGCCC__\n#include "cuda_ad.h"\n#include "utils.h"\n#endif')
    content = content.replace(
        '#include "cuda_ad.h"',
        '#ifdef __HGGCCC__\n#include "cuda_ad.h"\n#endif')
    return content


def process_file(filepath: str, repo_dir: str,
                 dry_run: bool = False, verbose: bool = False) -> int:
    """Process a single file through the replacement pipeline. Returns change count."""
    # realpath on both sides: main() chdir's into repo_dir, so a symlink anywhere in
    # the path would otherwise make relpath climb out and every rel_path key miss.
    rel_path = os.path.relpath(os.path.realpath(filepath), os.path.realpath(repo_dir))

    # --- setup.py ---
    if rel_path == 'setup.py':
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except (IOError, OSError):
            return 0
        if 'HGCCBuildExtension' in content:
            print("  WARNING: setup.py uses the legacy HGCCBuildExtension, which this "
                  "script no longer converts, left untouched")
            return 0
        if 'CUDAExtension' not in content:
            print("  WARNING: setup.py matches no known shape, left untouched")
            return 0
        new_content = rewrite_setup_py(content)
        if '-arch=ppu_' in new_content or 'hgcc' in new_content:
            print("  WARNING: setup.py still has PPU-only build flags after rewrite")
        if new_content == content:
            return 0
        if dry_run:
            if verbose:
                print(f"    [dry-run] {rel_path}")
            return 1
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        if verbose:
            print(f"    {rel_path}")
        return 1

    # --- Excluded files ---
    if rel_path in EXCLUDE_FILES:
        return 0

    # --- Read content ---
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
    except (IOError, OSError):
        return 0
    original = content

    # --- Check relevance ---
    has_file_specific = rel_path in FILE_SPECIFIC_REPLACEMENTS
    has_hggc_patterns = bool(re.search(
        r'\bhggc|\b__HGGC_ARCH__\b|\bPPU10_|__hg_fp8|__ppu_bfloat16|<hgtx3/|<hggc_',
        content))
    if not has_file_specific and not has_hggc_patterns:
        return 0

    # --- Pipeline ---
    # 1. File-specific replacements (highest priority)
    if has_file_specific:
        content = apply_replacements(content, FILE_SPECIFIC_REPLACEMENTS[rel_path])

    # 2. Category replacements (include → enum → type → func → macro → nvtx)
    for table in ALL_CATEGORY_REPLACEMENTS:
        content = apply_replacements(content, table)

    # 3. Word-boundary type replacements (FP8/bfloat16 — \b prevents prefix matches)
    content = apply_word_boundary_replacements(content, WORD_BOUNDARY_REPLACEMENTS)

    # 4. Arch value replacements (line-by-line on __CUDA_ARCH__ lines)
    content = apply_arch_value_replacements(content)

    # 5. Regex replacements (complex multi-line patterns)
    content = apply_regex_replacements(content, REGEX_REPLACEMENTS, filepath)

    # 6. __HGGCCC__ guards around cuda_ad.h (after <hggc_ad.h> → "cuda_ad.h")
    content = apply_hggccc_guards(content, rel_path)

    # --- Check and write ---
    if content == original:
        return 0

    if dry_run:
        if verbose:
            print(f"    [dry-run] {rel_path}")
        return 1

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    if verbose:
        print(f"    {rel_path}")
    return 1


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='FlashMLA PPU-original → CUDA-Compatible Transform (lookup-table refactored)')
    parser.add_argument('target_dir', nargs='?', default='.',
                        help='Target FlashMLA root directory')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without making changes')
    parser.add_argument('--verbose', action='store_true',
                        help='Print each modified file')
    args = parser.parse_args()

    target_dir = os.path.abspath(args.target_dir)
    if (not os.path.isfile(os.path.join(target_dir, 'setup.py')) or
            not os.path.isdir(os.path.join(target_dir, 'csrc'))):
        print(f"ERROR: {target_dir} is not a FlashMLA root directory.")
        sys.exit(1)
    os.chdir(target_dir)

    print("=" * 60)
    print("FlashMLA PPU-original → CUDA-Compatible Transform (lookup-table)")
    print(f"Target: {target_dir}")
    print(f"Mode: {'DRY-RUN' if args.dry_run else 'APPLY'}")
    print("=" * 60)

    t0 = time.time()

    # Collect all target files
    all_files = []
    for root, dirs, files in os.walk('csrc'):
        if 'actlize' in root:
            continue
        for f in files:
            if f.endswith(('.cu', '.cuh', '.h', '.hpp', '.cpp')):
                all_files.append(os.path.join(root, f))

    # Also include setup.py
    setup_path = os.path.join(target_dir, 'setup.py')
    if os.path.isfile(setup_path):
        all_files.append(setup_path)

    print(f"\nFound {len(all_files)} target files to process\n")

    # Process all files through the unified pipeline
    files_modified = 0
    for fp in sorted(all_files):
        changes = process_file(fp, target_dir, dry_run=args.dry_run, verbose=args.verbose)
        if changes > 0:
            files_modified += 1

    # Summary
    t1 = time.time()
    print(f"\nDone in {t1 - t0:.1f}s")
    print("=" * 60)
    print(f"  Files scanned:   {len(all_files)}")
    print(f"  Files modified:  {files_modified}")
    print("=" * 60)

    if args.dry_run:
        print("\n  [DRY RUN — no files modified]")
    else:
        print("\n  Next steps:")
        print("  1. Build: python setup.py bdist_wheel")
        print("  2. Install: pip install dist/flash_mla-*.whl --force-reinstall --no-deps")
        print("  3. Test: python -m pytest tests/test_flash_mla.py -v")


if __name__ == '__main__':
    main()
