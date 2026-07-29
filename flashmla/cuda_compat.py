#!/usr/bin/env python3
"""
FlashMLA PPU-original → CUDA-Compatible Transformation Script.

Transforms FlashMLA from ppu-original compilation back to
nvcc wrapper / CUDAExtension compilation.

Usage:
    python mla_compat.py [TARGET_DIR] [--dry-run] [--verbose]

Run from the FlashMLA root directory, or pass it as TARGET_DIR.
Fully self-contained — no git, no subprocess, no external dependencies.
"""

import argparse
import os
import re
import sys
import time
from datetime import datetime
# =============================================================================
# Reverse replacement maps (hggc/PPU → cuda/SM) — applied as regex word-boundary
# =============================================================================

REVERSE_REPLACEMENTS = [
    # --- Runtime API (longer patterns first) ---
    (r'\bhggcOccupancyMaxActiveBlocksPerMultiprocessor\b', 'cudaOccupancyMaxActiveBlocksPerMultiprocessor'),
    (r'\bhggcFuncAttributeMaxDynamicSharedMemorySize\b', 'cudaFuncAttributeMaxDynamicSharedMemorySize'),
    (r'\bhggcFuncAttributes\b', 'cudaFuncAttributes'),
    (r'\bhggcFuncAttribute\b', 'cudaFuncAttribute'),
    (r'\bhggcTriggerProgrammaticLaunchCompletion\b', 'cudaTriggerProgrammaticLaunchCompletion'),
    (r'\bhggcGridDependencySynchronize\b', 'cudaGridDependencySynchronize'),
    (r'\bhggcDeviceGetAttribute\b', 'cudaDeviceGetAttribute'),
    (r'\bhggcDevAttrMaxSharedMemoryPerMultiprocessor\b', 'cudaDevAttrMaxSharedMemoryPerMultiprocessor'),
    (r'\bhggcDevAttrMaxSharedMemoryPerBlockOptin\b', 'cudaDevAttrMaxSharedMemoryPerBlockOptin'),
    (r'\bhggcDevAttrMultiProcessorCount\b', 'cudaDevAttrMultiProcessorCount'),
    (r'\bhggcDevAttrComputeCapabilityMajor\b', 'cudaDevAttrComputeCapabilityMajor'),
    (r'\bhggcDevAttrComputeCapabilityMinor\b', 'cudaDevAttrComputeCapabilityMinor'),
    (r'\bhggcOccupancyMaxActiveBlocksPerMultiprocessor\b', 'cudaOccupancyMaxActiveBlocksPerMultiprocessor'),
    (r'\bhggcGetFuncBySymbol\b', 'cudaGetFuncBySymbol'),
    (r'\bhggcFuncSetAttribute\b', 'cudaFuncSetAttribute'),
    (r'\bhggcFuncGetAttributes\b', 'cudaFuncGetAttributes'),
    (r'\bhggcGetErrorString\b', 'cudaGetErrorString'),
    (r'\bhggcGetLastError\b', 'cudaGetLastError'),
    (r'\bhggcGetDevice\b', 'cudaGetDevice'),
    (r'\bhggcLaunchKernelExAD\b', 'cuLaunchKernelExAD'),
    (r'\bhggcLaunchKernelExC\b', 'cudaLaunchKernelEx'),
    (r'\bhggcLaunchKernelEx\b', 'cudaLaunchKernelEx'),

    # --- Launch config types ---
    (r'\bhggcLaunchAttributeProgrammaticStreamSerialization\b', 'cudaLaunchAttributeProgrammaticStreamSerialization'),
    (r'\bhggcLaunchConfig_t\b', 'cudaLaunchConfig_t'),
    (r'\bhggcLaunchAttribute\b', 'cudaLaunchAttribute'),
    (r'\bhggcLaunchKernel\b', 'cudaLaunchKernel'),

    # --- Stream capture / sync API ---
    (r'\bhggcStreamCaptureModeRelaxed\b', 'cudaStreamCaptureModeRelaxed'),
    (r'\bhggcStreamCaptureModeNone\b', 'cudaStreamCaptureStatusNone'),
    (r'\bhggcStreamCaptureStatusNone\b', 'cudaStreamCaptureStatusNone'),
    (r'\bhggcStreamCaptureStatus\b', 'cudaStreamCaptureStatus'),
    (r'\bhggcStreamCaptureMode\b', 'cudaStreamCaptureMode'),
    (r'\bhggcStreamIsCapturing\b', 'cudaStreamIsCapturing'),
    (r'\bhggcStreamCreateWithFlags\b', 'cudaStreamCreateWithFlags'),
    (r'\bhggcStreamDestroy\b', 'cudaStreamDestroy'),
    (r'\bhggcStreamSynchronize\b', 'cudaStreamSynchronize'),
    (r'\bhggcThreadExchangeStreamCaptureMode\b', 'cudaThreadExchangeStreamCaptureMode'),
    (r'\bhggcStreamNonBlocking\b', 'cudaStreamNonBlocking'),
    (r'\bhggcStreamDefault\b', 'cudaStreamDefault'),
    (r'\bhggcStream_t\b', 'cudaStream_t'),

    # --- Memory API ---
    (r'\bhggcMemcpyDeviceToHost\b', 'cudaMemcpyDeviceToHost'),
    (r'\bhggcMemcpyDeviceToDevice\b', 'cudaMemcpyDeviceToDevice'),
    (r'\bhggcMemcpyHostToDevice\b', 'cudaMemcpyHostToDevice'),
    (r'\bhggcMemcpyHostToHost\b', 'cudaMemcpyHostToHost'),
    (r'\bhggcMemcpyKind\b', 'cudaMemcpyKind'),
    (r'\bhggcMemcpyAsync\b', 'cudaMemcpyAsync'),
    (r'\bhggcMemcpy\b', 'cudaMemcpy'),
    (r'\bhggcMemsetAsync\b', 'cudaMemsetAsync'),
    (r'\bhggcMalloc\b', 'cudaMalloc'),
    (r'\bhggcFree\b', 'cudaFree'),
    (r'\bhggcMemGetInfo\b', 'cudaMemGetInfo'),
    (r'\bhggcDeviceSynchronize\b', 'cudaDeviceSynchronize'),
    (r'\bhggcSetDevice\b', 'cudaSetDevice'),
    (r'\bhggcGetDeviceProperties\b', 'cudaGetDeviceProperties'),
    (r'\bhggcGetErrorName\b', 'cudaGetErrorName'),
    (r'\bhggcPeekAtLastError\b', 'cudaPeekAtLastError'),
    (r'\bhggcErrorUnknown\b', 'cudaErrorUnknown'),
    (r'\bhggcOccupancyMaxPotentialBlockSize\b', 'cudaOccupancyMaxPotentialBlockSize'),
    (r'\bhggcOccupancyDisableCachingOverride\b', 'cudaOccupancyDisableCachingOverride'),

    # --- Runtime types ---
    (r'\bhggcFunction_t\b', 'cudaFunction_t'),
    (r'\bhggcError_t\b', 'cudaError_t'),
    (r'\bhggcError\b', 'cudaError'),
    (r'\bhggcSuccess\b', 'cudaSuccess'),
    (r'\bhggcDataType_t\b', 'cudaDataType_t'),
    (r'\bhggcEvent_t\b', 'cudaEvent_t'),
    (r'\bhggcEventCreate\b', 'cudaEventCreate'),
    (r'\bhggcEventDestroy\b', 'cudaEventDestroy'),
    (r'\bhggcEventElapsedTime\b', 'cudaEventElapsedTime'),
    (r'\bhggcEventRecord\b', 'cudaEventRecord'),
    (r'\bhggcEventSynchronize\b', 'cudaEventSynchronize'),

    # --- Driver API types ---
    (r'\bHGlaunchAttributeAD\b', 'CUlaunchAttributeAD'),
    (r'\bHGlaunchConfigAD\b', 'CUlaunchConfigAD'),
    (r'\bHGAD_LAUNCH_ATTRIBUTE_IGNORE\b', 'CUAD_LAUNCH_ATTRIBUTE_IGNORE'),
    (r'\bHGfunction\b', 'CUfunction'),
    (r'\bHGresult\b', 'CUresult'),
    (r'\bHGGC_SUCCESS\b', 'CUDA_SUCCESS'),
    (r'\bhgLaunchKernelExAD\b', 'cuLaunchKernelExAD'),
    (r'\bhgGetErrorName\b', 'cuGetErrorName'),
    (r'\bhgGetErrorString\b', 'cuGetErrorString'),

    # --- Headers ---
    (r'<hggc_runtime\.h>', '<cuda_runtime.h>'),
    (r'"hggc_runtime\.h"', '"cuda_runtime.h"'),
    (r'<hggc_fp16\.h>', '<cuda_fp16.h>'),
    (r'<hggc_bf16\.h>', '<cuda_bf16.h>'),
    (r'<hggc_pipeline\.h>', '<cuda_pipeline.h>'),
    (r'<hggc_awbarrier\.h>', '<cuda_awbarrier.h>'),
    (r'<hggc_ad\.h>', '"cuda_ad.h"'),

    # --- NVTX / HGTX ---
    (r'<hgtx3/hgToolsExt\.h>', '<nvtx3/nvToolsExt.h>'),
    (r'\bhgtxEventAttributes_t\b', 'nvtxEventAttributes_t'),
    (r'\bhgtxDomainHandle_t\b', 'nvtxDomainHandle_t'),
    (r'\bhgtxDomainCreateA\b', 'nvtxDomainCreateA'),
    (r'\bhgtxDomainDestroy\b', 'nvtxDomainDestroy'),
    (r'\bhgtxDomainRangePushEx\b', 'nvtxDomainRangePushEx'),
    (r'\bhgtxDomainRangePop\b', 'nvtxDomainRangePop'),
    (r'\buse_hgtx_\b', 'use_nvtx_'),
    (r'\bHGTX_VERSION\b', 'NVTX_VERSION'),
    (r'\bHGTX_MESSAGE_TYPE_ASCII\b', 'NVTX_MESSAGE_TYPE_ASCII'),

    # --- Architecture macro ---
    (r'__HGGC_ARCH__', '__CUDA_ARCH__'),

    # --- FP8 types ---
    (r'\b__hg_fp8_e8m0\b', '__nv_fp8_e8m0'),
    (r'\b__hg_fp8x4_e4m3\b', '__nv_fp8x4_e4m3'),
    (r'\b__ppu_bfloat162\b', '__nv_bfloat162'),
    (r'\b__ppu_bfloat16_raw\b', '__nv_bfloat16_raw'),
    (r'\b__ppu_bfloat16\b', '__nv_bfloat16'),

    # Note: __HGGCCC__ is NOT converted — ppu_dev uses __HGGCCC__ for AIU launch guards.
    # In cuda-compat mode (nvcc), __HGGCCC__ is defined by PPU SDK's nvcc wrapper.
]

# Arch value reversals: applied only on lines containing __CUDA_ARCH__
# PPU __HGGC_ARCH__ >= 100 maps to __CUDA_ARCH__ >= 800 for sm_80 features (MMA)
# and __CUDA_ARCH__ >= 750 for sm_75 features (LDSM). The context-aware fix
# for LDSM is handled in apply_reverse_arch_values().
REVERSE_ARCH_VALUES = [
    ('>= 100', '>= 800'),
    ('== 100', '== 800'),
    ('== 150', '== 890'),
]


def apply_reverse_replacements(content):
    """Apply all reverse replacement regexes."""
    for pattern, replacement in REVERSE_REPLACEMENTS:
        content = re.sub(pattern, replacement, content)
    return content


def apply_reverse_arch_values(content):
    """Replace arch values only on lines containing __CUDA_ARCH__.
    
    Special case: lines followed by LDSM/SmemCopyAtom use >= 750 (sm_75),
    not >= 800 (sm_80). This matches ppu_dev's conditional structure.
    """
    lines = content.split('\n')
    new_lines = []
    for i, line in enumerate(lines):
        if '__CUDA_ARCH__' in line:
            for old, new in REVERSE_ARCH_VALUES:
                line = line.replace(old, new)
            # Fix: LDSM section uses >= 750, not >= 800
            # Check next line for LDSM or SmemCopyAtom keywords
            next_line = lines[i + 1] if i + 1 < len(lines) else ''
            if 'LDSM' in next_line or 'SmemCopyAtom' in next_line:
                line = line.replace('>= 800', '>= 750')
        new_lines.append(line)
    return '\n'.join(new_lines)


# =============================================================================
# Structural file restorations (no git needed)
# =============================================================================

def restore_flash_h(dry_run=False, verbose=False):
    """Restore csrc/flash.h: remove #ifdef __HGGCCC__ blocks, restore cuda types."""
    fp = 'csrc/flash.h'
    if not os.path.isfile(fp):
        return False
    content = open(fp).read()
    if 'hggcStream_t' not in content and '__HGGCCC__' not in content:
        return False

    if dry_run:
        if verbose:
            print(f"    [dry-run] {fp}")
        return True

    # 1a. Replace the #ifdef __HGGCCC__ include block at top (old format)
    old_include = re.compile(
        r'#ifdef __HGGCCC__\n.*?#endif\n',
        re.DOTALL)
    content = old_include.sub('#include <ATen/cuda/CUDAContext.h>\n', content, count=1)

    # 1b. Replace direct hggc_runtime.h + typedef (ppu original format)
    content = re.sub(
        r'#include <hggc_runtime\.h>\n.*?typedef struct HGstream_st\* hggcStream_t;\n',
        '#include <ATen/cuda/CUDAContext.h>\n',
        content, flags=re.DOTALL)

    # 2a. Remove #ifdef __HGGCCC__ in is_sm89_or_newer(), keep only the #else branch
    old_func = re.compile(
        r'static bool is_sm89_or_newer\(\)\{\n'
        r'#ifdef __HGGCCC__\n'
        r'.*?'
        r'#else\n'
        r'(.*?)'
        r'#endif\n'
        r'\}',
        re.DOTALL)
    content = old_func.sub(
        r'static bool is_sm89_or_newer(){\n\1}',
        content)

    # 2b. Replace direct hggc API in is_sm89_or_newer() (ppu-original format)
    content = re.sub(
        r'static bool is_sm89_or_newer\(\)\{\n'
        r'    int device = 0;\n'
        r'    hggcGetDevice\(&device\);\n'
        r'    int major = 0, minor = 0;\n'
        r'    hggcDeviceGetAttribute\(&major, hggcDevAttrComputeCapabilityMajor, device\);\n'
        r'    hggcDeviceGetAttribute\(&minor, hggcDevAttrComputeCapabilityMinor, device\);\n'
        r'    return \(major > 8\) \|\| \(major == 8 && minor >= 9\);\n'
        r'\}',
        'static bool is_sm89_or_newer(){\n'
        '    auto dprops = at::cuda::getCurrentDeviceProperties();\n'
        '    return (dprops->major > 8) || (dprops->major == 8 && dprops->minor >= 9);\n'
        '}',
        content)

    # 3. Bulk replace remaining hggc types
    content = apply_reverse_replacements(content)
    content = apply_reverse_arch_values(content)

    open(fp, 'w').write(content)
    return True


def restore_hardware_info_h(dry_run=False, verbose=False):
    """Restore csrc/hardware_info.h to original cuda-compat version."""
    fp = 'csrc/hardware_info.h'
    if not os.path.isfile(fp):
        return False
    content = open(fp).read()
    if 'hggc' not in content and '__HGGCCC__' not in content:
        return False

    if dry_run:
        if verbose:
            print(f"    [dry-run] {fp}")
        return True

    new_content = """\
/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <tuple>

#if !defined(__CUDACC_RTC__)
#include "cuda_runtime.h"
#endif

#define CHECK_CUDA(call)                                                       \\
  do {                                                                         \\
    cudaError_t status_ = call;                                                \\
    if (status_ != cudaSuccess) {                                              \\
      fprintf(stderr, "CUDA error (%s:%d): %s\\n", __FILE__, __LINE__,          \\
              cudaGetErrorString(status_));                                    \\
      exit(1);                                                                 \\
    }                                                                          \\
  } while (0)


inline int get_current_device() {
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    return device;
}

inline std::tuple<int, int> get_compute_capability(int device) {
    int capability_major, capability_minor;
    CHECK_CUDA(cudaDeviceGetAttribute(&capability_major, cudaDevAttrComputeCapabilityMajor, device));
    CHECK_CUDA(cudaDeviceGetAttribute(&capability_minor, cudaDevAttrComputeCapabilityMinor, device));
    return {capability_major, capability_minor};
}

inline int get_num_sm(int device) {
    int multiprocessor_count;
    CHECK_CUDA(cudaDeviceGetAttribute(&multiprocessor_count, cudaDevAttrMultiProcessorCount, device));
    return multiprocessor_count;
}
"""
    open(fp, 'w').write(new_content)
    return True


def restore_utils_h(dry_run=False, verbose=False):
    """Restore csrc/utils.h: fp16/bf16 includes, ATen guard, CUDA_DRIVER_CHECK."""
    fp = 'csrc/utils.h'
    if not os.path.isfile(fp):
        return False
    content = open(fp).read()
    if '__HGGCCC__' not in content and 'hggc_fp16' not in content:
        return False

    if dry_run:
        if verbose:
            print(f"    [dry-run] {fp}")
        return True

    # 1. Replace hggc fp16/bf16 includes with cuda versions
    content = content.replace(
        '#include <hggc_fp16.h>\n#include <hggc_bf16.h>',
        '#include <cuda_fp16.h>\n\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800\n#include <cuda_bf16.h>\n#endif')

    # 2. Remove #ifndef __HGGCCC__ guard around ATen include
    content = content.replace(
        '#ifndef __HGGCCC__\n#include <ATen/cuda/CUDAContext.h>\n#endif',
        '#include <ATen/cuda/CUDAContext.h>')

    # 3. Replace the dual CUDA_DRIVER_CHECK block with original
    old_driver = re.compile(
        r'#ifdef __HGGCCC__\n'
        r'#define CUDA_DRIVER_CHECK\(expr\).*?'
        r'#else\n'
        r'(.*?)'
        r'#endif\n',
        re.DOTALL)
    content = old_driver.sub(r'\1', content, count=1)

    # 4. Fix USE_PPU guard — ppu_dev uses commented-out original + simplified guard
    content = content.replace(
        '#if defined(USE_PPU) && ACOMPUTE_VERSION == 10000',
        '// #if defined(USE_PPU) && ACOMPUTE_VERSION == 10000\n#if defined USE_PPU')

    # 5. Fix CUDA_DRIVER_CHECK macro — ppu_dev uses TORCH_CHECK instead of printf
    content = content.replace(
        'printf("HG driver error: %s: %s\\n",                \\\n               (_name ? _name : "?"), (_str ? _str : "?")); \\\n    }',
        'TORCH_CHECK(false, "CUDA driver error ",            \\\n        (_name ? _name : "?"), ": ", (_str ? _str : "?"));  \\\n    }                                                       \\\n')

    # 5. Bulk replace remaining hggc symbols (HGresult, HGGC_SUCCESS, etc.)
    content = apply_reverse_replacements(content)

    open(fp, 'w').write(content)
    return True


def restore_launch_template_guards(dry_run=False, verbose=False):
    """Remove #ifdef __HGGCCC__ guards from flash_fwd_launch_template.h, restore c10 includes."""
    fp = 'csrc/flash_fwd_launch_template.h'
    if not os.path.isfile(fp):
        return False
    content = open(fp).read()
    if '#ifndef __HGGCCC__' not in content and '#ifdef __HGGCCC__' not in content:
        return False

    if dry_run:
        if verbose:
            print(f"    [dry-run] {fp}")
        return True

    # 1. Replace the include guard
    content = content.replace(
        '#ifndef __HGGCCC__\n#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK\n'
        '#include <ATen/cuda/CUDAContext.h>\n'
        '#else\n'
        '#define C10_CUDA_CHECK(x) (void)(x)\n'
        '#define C10_CUDA_KERNEL_LAUNCH_CHECK()\n'
        '#endif',
        '#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK\n'
        '#include <ATen/cuda/CUDAContext.h>')

    # 2. Replace #ifdef __HGGCCC__ / #include <hggc_ad.h> / #else / #include "cuda_ad.h" / #endif
    content = content.replace(
        '#ifdef __HGGCCC__\n#include <hggc_ad.h>\n#else\n#include "cuda_ad.h"\n#endif',
        '#include "cuda_ad.h"')

    # 3. Keep #ifdef __HGGCCC__ AIU launch blocks as-is (ppu_dev preserves them)
    # Only bulk-replace symbols inside them (HGfunction → CUfunction, etc.)

    # 4. Bulk replace remaining hggc types
    content = apply_reverse_replacements(content)
    content = apply_reverse_arch_values(content)

    # 5. Restore #ifdef __HGGCCC__ guard around cuda_ad.h + utils.h
    content = content.replace(
        '#include "cuda_ad.h"\n#include "utils.h"',
        '#ifdef __HGGCCC__\n#include "cuda_ad.h"\n#include "utils.h"\n#endif')

    # 6. Fix SM detection: get_num_sm(get_current_device()) → at::cuda::getCurrentDeviceProperties()
    # ppu_dev uses dprops->multiProcessorCount with ternary for 64→20
    content = re.sub(
        r'int sm_count = get_num_sm\(get_current_device\(\)\);\n\s*if \(sm_count == 64\) sm_count = 20;',
        'auto dprops = at::cuda::getCurrentDeviceProperties();\n            int sm_count = dprops->multiProcessorCount == 64 ? 20 : dprops->multiProcessorCount;',
        content)

    # 7. Fix C10_CUDA_CHECK wrapper for cudaFuncSetAttribute
    content = re.sub(
        r'(\s*)cudaFuncSetAttribute\(\n\s*(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);',
        r'\1C10_CUDA_CHECK(cudaFuncSetAttribute(\n\2));',
        content)

    # 8. Remove #ifdef __HGGCCC__ warp_interleave block — ppu_dev uses only the #else path
    content = re.sub(
        r'#ifdef __HGGCCC__\n\s*// Under hgcc.*?#else\n\s*\{\n\s*auto dprops = at::cuda::getCurrentDeviceProperties\(\);\n\s*if \(std::string\(dprops->name\)\.find\("610"\) != std::string::npos\)\n\s*warp_interleave = false;\n\s*\}\n#endif\n',
        '    auto dprops = at::cuda::getCurrentDeviceProperties();\n    if (std::string(dprops->name).find("610") != std::string::npos)\n        warp_interleave = false;\n',
        content, flags=re.DOTALL)

    open(fp, 'w').write(content)
    return True


def restore_flash_api_cpp(dry_run=False, verbose=False):
    """Restore csrc/flash_api.cpp: reverse stream type casts and dprops replacements."""
    fp = 'csrc/flash_api.cpp'
    if not os.path.isfile(fp):
        return False
    content = open(fp).read()
    if 'hggcStream_t' not in content:
        return False

    if dry_run:
        if verbose:
            print(f"    [dry-run] {fp}")
        return True

    # Reverse: hggcStream_t stream = (hggcStream_t)at::cuda::getCurrentCUDAStream().stream();
    # → auto stream = at::cuda::getCurrentCUDAStream().stream();
    content = re.sub(
        r'hggcStream_t\s+stream\s*=\s*\(hggcStream_t\)(at::cuda::getCurrentCUDAStream\(\)\.stream\(\))',
        r'auto stream = \1', content)

    # Also handle: hggcStream_t stream = (hggcStream_t)params.stream;
    content = content.replace(
        '(hggcStream_t)params.stream', 'params.stream')

    # Bulk replace remaining types
    content = apply_reverse_replacements(content)

    # Remove redundant (cudaStream_t) casts from function calls
    content = re.sub(r'\(cudaStream_t\)', '', content)

    open(fp, 'w').write(content)
    return True


def restore_traits_h(dry_run=False, verbose=False):
    """Restore csrc/flash_splitkv/traits.h: pipeline/awbarrier headers + atom names."""
    fp = 'csrc/flash_splitkv/traits.h'
    if not os.path.isfile(fp):
        return False
    content = open(fp).read()
    if 'hggc_pipeline' not in content and 'PPU10_' not in content:
        return False

    if dry_run:
        if verbose:
            print(f"    [dry-run] {fp}")
        return True

    content = apply_reverse_replacements(content)
    content = apply_reverse_arch_values(content)

    open(fp, 'w').write(content)
    return True


def restore_metadata_cu(dry_run=False, verbose=False):
    """Restore csrc/flash_fwd_mla_metadata.cu."""
    fp = 'csrc/flash_fwd_mla_metadata.cu'
    if not os.path.isfile(fp):
        return False
    content = open(fp).read()
    if 'hggcStream_t' not in content:
        return False

    if dry_run:
        if verbose:
            print(f"    [dry-run] {fp}")
        return True

    content = apply_reverse_replacements(content)
    open(fp, 'w').write(content)
    return True


def restore_splitkv_files(dry_run=False, verbose=False):
    """Restore csrc/flash_splitkv/*.cu and *.h files."""
    files = [
        'csrc/flash_splitkv/splitkv_mla.cu',
        'csrc/flash_splitkv/splitkv_mla.h',
        'csrc/flash_splitkv/splitkv_dsa.cu',
        'csrc/flash_splitkv/mla_combine.cu',
        'csrc/flash_splitkv/mla_combine.h',
    ]
    count = 0
    for fp in files:
        if not os.path.isfile(fp):
            continue
        content = open(fp).read()
        if 'hggcStream_t' not in content and 'hggcFuncSetAttribute' not in content \
           and '__HGGC_ARCH__' not in content and 'hggc' not in content \
           and 'hgtx' not in content:
            continue

        if dry_run:
            if verbose:
                print(f"    [dry-run] {fp}")
            count += 1
            continue

        content = apply_reverse_replacements(content)
        content = apply_reverse_arch_values(content)
        # Fix mla_combine.cu: cudaLaunchKernelEx(args) with void* args → direct args
        # ppu_dev uses cudaLaunchKernelEx(&config, kernel, params), not ExC with void**
        content = re.sub(
            r'cudaLaunchKernelEx\(([^,]+),\s*\(const void\*\)([^,]+),\s*\(void\*\*\)&([^)]+)\)',
            r'cudaLaunchKernelEx(\1, \2, \3)', content)
        # Restore #ifdef __HGGCCC__ guard around cuda_ad.h + utils.h in mla_combine.cu
        content = content.replace(
            '#include "cuda_ad.h"\n#include "utils.h"',
            '#ifdef __HGGCCC__\n#include "cuda_ad.h"\n#include "utils.h"\n#endif')
                
        # Fix SM detection + C10_CUDA_CHECK: only for splitkv_mla.cu and splitkv_dsa.cu
        if 'splitkv_mla' in fp or 'splitkv_dsa' in fp:
            content = re.sub(
                r'int sm_count = get_num_sm\(get_current_device\(\)\);\n\s*if \(sm_count == 64\) sm_count = 20;',
                'auto dprops = at::cuda::getCurrentDeviceProperties();\n\n            int sm_count = dprops->multiProcessorCount;\n            if (std::string(dprops->name).find("810E") != std::string::npos) {\n                sm_count = 20;\n            }',
                content)
        
            # Fix C10_CUDA_CHECK wrapper: ppu_dev wraps cudaFuncSetAttribute with C10_CUDA_CHECK
            content = re.sub(
                r'(\s*)cudaFuncSetAttribute\(',
                r'\1C10_CUDA_CHECK(cudaFuncSetAttribute(', content)
            content = re.sub(
                r'C10_CUDA_CHECK\(cudaFuncSetAttribute\(([^;]+)\);',
                r'C10_CUDA_CHECK(cudaFuncSetAttribute(\1));', content)
        
        open(fp, 'w').write(content)
        count += 1
        if verbose:
            print(f"    {fp}")
    return count


def restore_cu_files(dry_run=False, verbose=False):
    """Restore main .cu kernel files."""
    files = [
        'csrc/flash_fwd_split_hdim576_512_bf16_sm80.cu',
        'csrc/flash_fwd_split_hdim576_512_fp16_sm80.cu',
        'csrc/flash_fwd_sparse_prefill_hdim576_512_bf16_sm80.cu',
    ]
    count = 0
    for fp in files:
        if not os.path.isfile(fp):
            continue
        content = open(fp).read()
        if 'hggcStream_t' not in content and 'hggcFuncSetAttribute' not in content:
            continue

        if dry_run:
            if verbose:
                print(f"    [dry-run] {fp}")
            count += 1
            continue

        content = apply_reverse_replacements(content)
        content = apply_reverse_arch_values(content)
        open(fp, 'w').write(content)
        count += 1
        if verbose:
            print(f"    {fp}")
    return count


def restore_other_headers(dry_run=False, verbose=False):
    """Restore remaining headers: kernel_traits.h, dequant.h, acc_vreg_fraga.h, flash.h."""
    files = [
        'csrc/kernel_traits.h',
        'csrc/dequant.h',
        # Note: acc_vreg_fraga.h is NOT processed — ppu_dev uses __HGGC_ARCH__
        # in this file (same as ppu-original), so no conversion needed.
        'csrc/fmha_profiling_interface.hpp',
    ]
    count = 0
    for fp in files:
        if not os.path.isfile(fp):
            continue
        content = open(fp).read()
        if '__HGGC_ARCH__' not in content and 'PPU10_' not in content \
           and 'PPU_8x' not in content and 'PPU0015_' not in content \
           and 'PPU0010_' not in content and '__hg_fp8' not in content \
           and 'hggcStream_t' not in content \
           and 'hgtx' not in content and 'hggc' not in content:
            continue

        if dry_run:
            if verbose:
                print(f"    [dry-run] {fp}")
            count += 1
            continue

        content = apply_reverse_replacements(content)
        content = apply_reverse_arch_values(content)

        # kernel_traits.h: remove extra SmemCopyAtomQ/K fallback in #else branch
        if fp == 'csrc/kernel_traits.h':
            content = re.sub(
                r'(using SmemCopyAtomTransposed = Copy_Atom<DefaultCopy, elem_type>;\n)'
                r'    using SmemCopyAtomQ = SmemCopyAtom;\n'
                r'    using SmemCopyAtomK = SmemCopyAtom;\n',
                r'\1', content)

        open(fp, 'w').write(content)
        count += 1
        if verbose:
            print(f"    {fp}")
    return count


# =============================================================================
# setup.py restoration (hardcode original CUDAExtension version)
# =============================================================================

ORIGINAL_SETUP_PY = r'''import os
from pathlib import Path
from datetime import datetime
import subprocess

from setuptools import setup, find_packages

from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
    IS_WINDOWS,
)

DISABLE_FP16 = os.getenv("FLASH_MLA_DISABLE_FP16", "FALSE") == "TRUE"
ENABLE_C_DECODE_SPARSE = os.getenv("FLASHMLA_C_ENABLE_DECODE_SPARSE", "TRUE") == "TRUE"
CPP_INFERENCE = 'FLASH_MLA_CPP_INFER_BUILD' in os.environ.keys() and os.environ['FLASH_MLA_CPP_INFER_BUILD'] == "1"

def append_nvcc_threads(nvcc_extra_args):
    nvcc_threads = os.getenv("NVCC_THREADS") or "32"
    return nvcc_extra_args + ["--threads", nvcc_threads]


def get_sources():
    sources = [
        "csrc/flash_api.cpp",
        "csrc/flash_fwd_split_hdim576_512_bf16_sm80.cu",
        "csrc/flash_fwd_sparse_prefill_hdim576_512_bf16_sm80.cu",
        "csrc/flash_fwd_mla_metadata.cu",
        "csrc/flash_splitkv/mla_combine.cu",
        "csrc/flash_splitkv/splitkv_mla.cu",
        "csrc/flash_splitkv/splitkv_dsa.cu",
    ]

    if not DISABLE_FP16:
        sources.append("csrc/flash_fwd_split_hdim576_512_fp16_sm80.cu")

    return sources

def get_features_args():
    features_args = []
    if DISABLE_FP16:
        features_args.append("-DFLASH_MLA_DISABLE_FP16")
    if ENABLE_C_DECODE_SPARSE:
        features_args.append("-DFLASHMLA_C_ENABLE_DECODE_SPARSE")
    if CPP_INFERENCE:
        features_args.append("-DFLASH_MLA_CPP_INFER_BUILD")
    features_args.append("-DFLASH_MLA_STANDALONE_BUILD")
    features_args.append("-DUSE_TS")

    return features_args

this_dir = os.path.dirname(os.path.abspath(__file__))
# subprocess.run(["git", "submodule", "update", "--init", "csrc/actlize"])  # disabled: no git
dir_actlize = this_dir +  "/csrc/actlize"
# check the existence of actlize
if not os.path.exists(dir_actlize):
    try:
        repo_actlize = os.path.dirname(this_dir) + "/actlize"
        if not os.path.exists(repo_actlize):
            raise RuntimeError(
                f"actlize does not exist: actlize must be fetched in advance as:\n"
                f" \"{repo_actlize}\" or \"{dir_actlize}\""
            )
        else:
            os.symlink(repo_actlize, dir_actlize)
    except Exception as e:
        raise EnvironmentError("setup dependencies FAILED: " + repr(e))

cc_flag = []
cc_flag.append("-gencode")
cc_flag.append("arch=compute_80,code=sm_80")

cxx_args = ["-O3", "-std=c++17", "-DNDEBUG", "-Wno-deprecated-declarations"]

ext_modules = []
ext_modules.append(
    CUDAExtension(
        name="flash_mla_cuda",
        sources=get_sources(),
        libraries=['cuda'],
        extra_compile_args={
            "cxx": cxx_args + get_features_args(),
            "nvcc": append_nvcc_threads(
                [
                    "-O3",
                    "-std=c++17",
                    "-DNDEBUG",
                    "-D_USE_MATH_DEFINES",
                    "-Wno-deprecated-declarations",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                    "--ptxas-options=-v,--register-usage-level=10",
                    "-mllvm",
                    "-ppu-max-vreg-count=256",
                    "-mllvm",
                    "-ppu-sink-matrix-addr=true",
                    "-mllvm",
                    "-ppu-max-alloca-byte-size=320",
                    "-mllvm",
                    "-ppu-sink-async-addr=true",
                    "-mllvm",
                    "-ppu-sink-load-addr=true",
                    "-mllvm",
                    "-ppu-sink-store-addr=true",
                    "-mllvm",
                    "-ppu-alloca-half-ldst-simplify=true",
                    "-mllvm",
                    "-ppu-force-warpage=true",
                    "-mllvm",
                    "-ppu-force-vregrr=true",
                    "-DUSE_PPU",
                    "-DUSE_AIU=1",
                    "-DACOMPUTE_VERSION=10000"
                ]
                + cc_flag
            ) + get_features_args(),
        },
        include_dirs=[
            Path(this_dir) / "csrc",
            Path(this_dir) / "csrc" / "actlize" / "include",
        ],
    )
)


try:
    _now = datetime.now()
    rev = '+' + _now.strftime("%Y-%m-%d-%H-%M-%S")
except Exception as _:
    rev = '+dev'

def custom_local_scheme(version):
    return '+dev%03d.%s' % (version.distance, version.node[:7])

def custom_version_scheme(version):
    return '2.0.0'

setup(
    name="flash_mla",
    use_scm_version={
        "local_scheme": custom_local_scheme,
        "version_scheme": custom_version_scheme,
    },
    setup_requires=["setuptools-scm==9.2.2"],
    packages=find_packages(include=['flash_mla']),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
'''


def restore_setup_py(dry_run=False, verbose=False):
    """Restore setup.py to original CUDAExtension version."""
    fp = 'setup.py'
    if not os.path.isfile(fp):
        return False
    content = open(fp).read()
    if 'HGCCBuildExtension' not in content:
        return False

    if dry_run:
        if verbose:
            print(f"    [dry-run] {fp}")
        return True

    open(fp, 'w').write(ORIGINAL_SETUP_PY)
    return True


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='FlashMLA PPU-original → CUDA-Compatible Transform')
    parser.add_argument('target_dir', nargs='?', default='.', help='Target FlashMLA root directory')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done')
    parser.add_argument('--verbose', action='store_true', help='Print each modified file')
    args = parser.parse_args()

    target_dir = os.path.abspath(args.target_dir)
    if not os.path.isfile(os.path.join(target_dir, 'setup.py')) or not os.path.isdir(os.path.join(target_dir, 'csrc')):
        print(f"ERROR: {target_dir} is not a FlashMLA root directory.")
        sys.exit(1)
    os.chdir(target_dir)

    print("=" * 60)
    print("FlashMLA PPU-original → CUDA-Compatible Transform")
    print(f"Target: {target_dir}")
    print("=" * 60)

    t0 = time.time()
    n_total = 0

    # Step 1: Structural restorations
    print("\n[1/3] Structural file restorations...")
    restorers = [
        ('csrc/flash.h', restore_flash_h),
        ('csrc/hardware_info.h', restore_hardware_info_h),
        ('csrc/utils.h', restore_utils_h),
        ('csrc/flash_fwd_launch_template.h', restore_launch_template_guards),
        ('csrc/flash_api.cpp', restore_flash_api_cpp),
        ('csrc/flash_splitkv/traits.h', restore_traits_h),
        ('csrc/flash_fwd_mla_metadata.cu', restore_metadata_cu),
    ]
    for name, func in restorers:
        if func(dry_run=args.dry_run, verbose=args.verbose):
            n_total += 1
            if not args.dry_run and args.verbose:
                print(f"    {name}")

    # Step 2: Bulk file transformations
    print("\n[2/3] Bulk file transformations...")
    n_bulk = restore_splitkv_files(dry_run=args.dry_run, verbose=args.verbose)
    n_bulk += restore_cu_files(dry_run=args.dry_run, verbose=args.verbose)
    n_bulk += restore_other_headers(dry_run=args.dry_run, verbose=args.verbose)
    n_total += n_bulk
    print(f"  Transformed: {n_bulk} files")

    # Step 3: setup.py restoration
    print("\n[3/3] Setup.py restoration...")
    if restore_setup_py(dry_run=args.dry_run, verbose=args.verbose):
        n_total += 1
        if not args.dry_run and args.verbose:
            print(f"    setup.py")

    # Summary
    t1 = time.time()
    print(f"\nDone in {t1-t0:.1f}s")
    print("=" * 60)
    print(f"  Total files modified: {n_total}")
    print("=" * 60)

    if args.dry_run:
        print("\n  [DRY RUN - no files modified]")
    else:
        print("\n  Next steps:")
        print("  1. Build: python setup.py bdist_wheel")
        print("  2. Install: pip install dist/flash_mla-*.whl --force-reinstall --no-deps")
        print("  3. Test: python -m pytest tests/test_flash_mla.py -v")


if __name__ == '__main__':
    main()
