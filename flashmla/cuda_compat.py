#!/usr/bin/env python3
"""
FlashMLA PPU-original → CUDA-Compatible Transformation Script.

Refactored with lookup-table approach for maintainability.
Transforms FlashMLA from ppu-original compilation back to CUDA-compatible source trees.

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
    ('hggcError_t', 'cudaError_t'),
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
]

# FP8 / bfloat16 type names require word-boundary matching so that e.g.
# __ppu_bfloat162 does not also match inside __ppu_bfloat162_raw.
WORD_BOUNDARY_REPLACEMENTS: List[Tuple[str, str]] = [
    ('__hg_fp8_e8m0', '__nv_fp8_e8m0'),
    ('__hg_fp8x4_e4m3', '__nv_fp8x4_e4m3'),
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

# Processing order: includ → enum → type → func → macro → nvtx
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

    # C10_CUDA_CHECK wrapping (multi-line)
    (r'([ \t]+)cudaFuncSetAttribute\(\n([ \t]+)(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size)\);',
     r'\1C10_CUDA_CHECK(cudaFuncSetAttribute(\n\2\3));', 'splitkv_mla.cuh'),
    (r'([ \t]+)cudaFuncSetAttribute\(\n([ \t]+)(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size)\);',
     r'\1C10_CUDA_CHECK(cudaFuncSetAttribute(\n\2\3));', 'splitkv_mla_kernel.cuh'),
    (r'([ \t]+)cudaFuncSetAttribute\(\n([ \t]+)(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size)\);',
     r'\1C10_CUDA_CHECK(cudaFuncSetAttribute(\n\2\3));', 'sparse_prefill_wg.cuh'),

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
        ('hggcError_t status_ = call;', 'cudaError_t status_ = call;'),
        ('if (status_ != hggcSuccess)', 'if (status_ != cudaSuccess)'),
        (
            'fprintf(stderr, "HGGC error (%s:%d): %s\\n", __FILE__, __LINE__,',
            'fprintf(stderr, "CUDA error (%s:%d): %s\\n", __FILE__, __LINE__,'
        ),
        ('hggcGetErrorString(status_)', 'cudaGetErrorString(status_)'),
        (
            '#define CHECK_CUDA_KERNEL_LAUNCH() CHECK_CUDA(hggcGetLastError())',
            '#define CHECK_CUDA_KERNEL_LAUNCH() CHECK_CUDA(cudaGetLastError())'
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
        # Replace GraphCaptureModeSuspender struct
        (
            'struct GraphCaptureModeSuspender {\n'
            '    hggcStreamCaptureMode original_mode;\n'
            '\n'
            '    explicit GraphCaptureModeSuspender(hggcStreamCaptureMode relaxed_mode = hggcStreamCaptureModeRelaxed) {\n'
            '        original_mode = relaxed_mode;\n'
            '        // Exchange current thread\'s mode with relaxed_mode, and store previous mode in original_mode\n'
            '        hggcThreadExchangeStreamCaptureMode(&original_mode);\n'
            '    }\n'
            '\n'
            '    ~GraphCaptureModeSuspender() {\n'
            '        // Restore the saved original mode back to the current thread\n'
            '        hggcThreadExchangeStreamCaptureMode(&original_mode);\n'
            '    }',
            'struct GraphCaptureModeSuspender {\n'
            '    cudaStreamCaptureMode original_mode;\n'
            '\n'
            '    explicit GraphCaptureModeSuspender(cudaStreamCaptureMode relaxed_mode = cudaStreamCaptureModeRelaxed) {\n'
            '        original_mode = relaxed_mode;\n'
            '        // Exchange current thread\'s mode with relaxed_mode, and store previous mode in original_mode\n'
            '        cudaThreadExchangeStreamCaptureMode(&original_mode);\n'
            '    }\n'
            '\n'
            '    ~GraphCaptureModeSuspender() {\n'
            '        // Restore the saved original mode back to the current thread\n'
            '        cudaThreadExchangeStreamCaptureMode(&original_mode);\n'
            '    }'
        ),
    ],

    # --- csrc/params.h ---
    'csrc/params.h': [
        (
            '#include <hggc_fp16.h>\n#include <hggc_runtime.h>',
            '#include <cuda_fp16.h>\n#include <cuda_runtime.h>\n#include <ATen/cuda/CUDAContext.h>'
        ),
        (
            '#include <hggc_bf16.h>',
            '#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800\n#include <cuda_bf16.h>\n#endif'
        ),
    ],

    # --- csrc/utils.h ---
    'csrc/utils.h': [
        (
            '#include <hggc_fp16.h>\n#include <hggc_bf16.h>',
            '#include <cuda_fp16.h>\n\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800\n#include <cuda_bf16.h>\n#endif'
        ),
        (
            '#define CUDA_DRIVER_CHECK(expr)                             \\\n'
            '    HGresult _r = (expr);                                   \\\n'
            '    if (_r != HGGC_SUCCESS) {                               \\\n'
            '        const char* _name = nullptr;                        \\\n'
            '        const char* _str = nullptr;                         \\\n'
            '        hgGetErrorName(_r, &_name);                         \\\n'
            '        hgGetErrorString(_r, &_str);                        \\\n'
            '        printf("HG driver error: %s: %s\\n",                \\\n'
            '               (_name ? _name : "?"), (_str ? _str : "?")); \\\n'
            '    }',
            '#define CUDA_DRIVER_CHECK(expr)                             \\\n'
            '    CUresult _r = (expr);                                   \\\n'
            '    if (_r != CUDA_SUCCESS) {                               \\\n'
            '        const char* _name = nullptr;                        \\\n'
            '        const char* _str = nullptr;                         \\\n'
            '        cuGetErrorName(_r, &_name);                         \\\n'
            '        cuGetErrorString(_r, &_str);                        \\\n'
            '        TORCH_CHECK(false, "CUDA driver error ",            \\\n'
            '        (_name ? _name : "?"), ": ", (_str ? _str : "?"));  \\\n'
            '    }'
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
# setup.py full content (hardcoded CUDAExtension version)
# =============================================================================

CUDA_SETUP_PY = r'''import os
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
        "csrc/api/api.cpp",
        "csrc/ppu/decode/dense/instantiations/hdim576_512_bf16.cu",
        "csrc/ppu/decode/dense/instantiations/splitkv_mla_bf16.cu",
        "csrc/ppu/prefill/sparse/instantiations/dispatch_bf16.cu",
        "csrc/ppu/decode/sparse/instantiations/hdim576_bf16.cu",
        "csrc/ppu/decode/sparse/instantiations/hdim512_bf16.cu",
        "csrc/ppuxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu",
        "csrc/ppuxx/decode/combine/instantiations/mla_combine_bf16.cu",
        "csrc/ppu/prefill/sparse/instantiations/wg_bf16_sm80.cu",
        "csrc/ppu/prefill/sparse/instantiations/wg_bf16_sm89.cu",
    ]

    if not DISABLE_FP16:
        sources.append("csrc/ppu/decode/dense/instantiations/hdim576_512_fp16.cu")
        sources.append("csrc/ppu/decode/dense/instantiations/splitkv_mla_fp16.cu")
        sources.append("csrc/ppuxx/decode/combine/instantiations/mla_combine_fp16.cu")
        sources.append("csrc/ppu/prefill/sparse/instantiations/wg_fp16_sm80.cu")
        sources.append("csrc/ppu/prefill/sparse/instantiations/wg_fp16_sm89.cu")

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
            Path(this_dir) / "csrc" / "api",
            Path(this_dir) / "csrc" / "kerutils" / "include",
            Path(this_dir) / "csrc" / "ppu",
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

### update FlashMLA Version
### flash_mla-1.0.0 do not support deepseek-v4, with fixed commit-id #f907f433e
### flash_mla-1.0.1 support deepseek-v4 with API Breaking Changes!
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

# =============================================================================
# Full file content: files that are completely rewritten (not incrementally patched)
# =============================================================================

FULL_FILE_CONTENT: Dict[str, str] = {
    'setup.py': CUDA_SETUP_PY,
    'csrc/kerutils/include/kerutils/host/hardware_info.h': (
        '/******************************************************************************\n'
        ' * Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD.\n'
        ' * Copyright (c) 2024, Tri Dao.\n'
        ' ******************************************************************************/\n'
        '\n'
        '#pragma once\n'
        '\n'
        '#include "kerutils/common/common.h"\n'
        '\n'
        '#include <tuple>\n'
        '\n'
        '#if !defined(__CUDACC_RTC__)\n'
        '#include "cuda_runtime.h"\n'
        '#endif\n'
        '\n'
        '\n'
        'inline int get_current_device() {\n'
        '    int device;\n'
        '    CHECK_CUDA(cudaGetDevice(&device));\n'
        '    return device;\n'
        '}\n'
        '\n'
        'inline std::tuple<int, int> get_compute_capability(int device) {\n'
        '    int capability_major, capability_minor;\n'
        '    CHECK_CUDA(cudaDeviceGetAttribute(&capability_major, cudaDevAttrComputeCapabilityMajor, device));\n'
        '    CHECK_CUDA(cudaDeviceGetAttribute(&capability_minor, cudaDevAttrComputeCapabilityMinor, device));\n'
        '    return {capability_major, capability_minor};\n'
        '}\n'
        '\n'
        'inline int get_num_sm(int device) {\n'
        '    int multiprocessor_count;\n'
        '    CHECK_CUDA(cudaDeviceGetAttribute(&multiprocessor_count, cudaDevAttrMultiProcessorCount, device));\n'
        '    return multiprocessor_count;\n'
        '}\n'
    ),
}

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
    rel_path = os.path.relpath(filepath, repo_dir)

    # --- Full file content rewrites ---
    if rel_path in FULL_FILE_CONTENT:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except (IOError, OSError):
            return 0
        target = FULL_FILE_CONTENT[rel_path]
        if content == target:
            return 0
        if dry_run:
            if verbose:
                print(f"    [dry-run] {rel_path}")
            return 1
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(target)
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

    # 2. Category replacements (include → enum → type → func → macro → atom → nvtx)
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
