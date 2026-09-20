#!/usr/bin/env python3
"""
Flex-Flash-Attention PPU-original -> CUDA-Compatible Transformation Script.

Variant of flash-attention/cuda_compat_v2.8.2.py with identical transform
rules; the only difference is that main() also walks the top-level
'flex_flash_attention/' module dir (see source_dirs), which the base
flash-attention converter does not cover.  Registered as the cudafy
'flex-flash-attention' subcommand.

Transforms flash-attention (FA2 + FA3) from ppu-original compilation back to
nvcc wrapper / CUDAExtension compilation mode. Pure regex/dict replacement,
no git dependency. Follows acompute's replace_cuda approach.

Usage:
    python3 cuda_compat_v2.8.2.py [TARGET_DIR] [--dry-run] [--verbose]

Run from the flash-attention root directory, or pass it as TARGET_DIR.
"""

import argparse
import os
import re
import sys
import time

# =============================================================================
# Reverse replacement maps (hggc/PPU -> cuda/SM)
# Sorted longest-first at runtime to avoid partial matches.
# =============================================================================

REVERSE_RUNTIME_API = {
    'hggcOccupancyMaxActiveBlocksPerMultiprocessor': 'cudaOccupancyMaxActiveBlocksPerMultiprocessor',
    'hggcFuncAttributeMaxDynamicSharedMemorySize': 'cudaFuncAttributeMaxDynamicSharedMemorySize',
    'hggcDevAttrMaxSharedMemoryPerMultiprocessor': 'cudaDevAttrMaxSharedMemoryPerMultiprocessor',
    'hggcDevAttrMaxSharedMemoryPerBlockOptin': 'cudaDevAttrMaxSharedMemoryPerBlockOptin',
    'hggcDevAttrComputeCapabilityMajor': 'cudaDevAttrComputeCapabilityMajor',
    'hggcDevAttrComputeCapabilityMinor': 'cudaDevAttrComputeCapabilityMinor',
    'hggcDevAttrMultiProcessorCount': 'cudaDevAttrMultiProcessorCount',
    'hggcDeviceGetAttribute': 'cudaDeviceGetAttribute',
    'hggcFuncGetAttributes': 'cudaFuncGetAttributes',
    'hggcFuncSetAttribute': 'cudaFuncSetAttribute',
    'hggcFuncAttributes': 'cudaFuncAttributes',
    'hggcGetErrorString': 'cudaGetErrorString',
    'hggcGetLastError': 'cudaGetLastError',
    'hggcGetDevice': 'cudaGetDevice',
    'hggcStream_t': 'cudaStream_t',
    'hggcError_t': 'cudaError_t',
    'hggcError': 'cudaError',
    'hggcSuccess': 'cudaSuccess',
    'hggcMemcpyDeviceToHost': 'cudaMemcpyDeviceToHost',
    'hggcErrorInvalidValue': 'cudaErrorInvalidValue',
    'hggcDeviceSynchronize': 'cudaDeviceSynchronize',
    'hggcMemsetAsync': 'cudaMemsetAsync',
    'hggcMemcpy': 'cudaMemcpy',
    'hggcMemset': 'cudaMemset',
    'hggcMalloc': 'cudaMalloc',
}

REVERSE_HEADER = {
    '<hggc.h>': '<cuda.h>',
    '"hggc.h"': '"cuda.h"',
    '<hggc_runtime.h>': '<cuda_runtime.h>',
    '"hggc_runtime.h"': '"cuda_runtime.h"',
    '<hggc_fp16.h>': '<cuda_fp16.h>',
    '<hggc_bf16.h>': '<cuda_bf16.h>',
    '<hgtx3/hgToolsExt.h>': '<nvtx3/nvToolsExt.h>',
}

REVERSE_MACRO = {
    '__HGGC_ARCH__': '__CUDA_ARCH__',
    '__HGGC_NO_HALF_OPERATORS__': '__CUDA_NO_HALF_OPERATORS__',
    '__HGGC_NO_HALF_CONVERSIONS__': '__CUDA_NO_HALF_CONVERSIONS__',
    '__HGGC_NO_HALF2_OPERATORS__': '__CUDA_NO_HALF2_OPERATORS__',
    '__HGGC_NO_BFLOAT16_CONVERSIONS__': '__CUDA_NO_BFLOAT16_CONVERSIONS__',
    'HGTX_VERSION': 'NVTX_VERSION',
    'HGTX_MESSAGE_TYPE_ASCII': 'NVTX_MESSAGE_TYPE_ASCII',
    'hgtxEventAttributes_t': 'nvtxEventAttributes_t',
    'hgtxDomainHandle_t': 'nvtxDomainHandle_t',
    'hgtxDomainCreateA': 'nvtxDomainCreateA',
    'hgtxDomainDestroy': 'nvtxDomainDestroy',
    'hgtxDomainRangePushEx': 'nvtxDomainRangePushEx',
    'hgtxDomainRangePop': 'nvtxDomainRangePop',
    'use_hgtx_': 'use_nvtx_',
}

# Applied only on lines containing __CUDA_ARCH__ (after macro replacement)
REVERSE_ARCH_VALUE = {
    '>= 100': '>= 800',
    '== 100': '== 800',
    '== 150': '== 890',
}

# Stream cast pattern: remove the (hggcStream_t) cast
STREAM_CAST_PATTERN = re.compile(
    r'hggcStream_t stream = \(hggcStream_t\)at::cuda::getCurrentCUDAStream\(\)\.stream\(\);')
STREAM_CAST_REPLACEMENT = 'auto stream = at::cuda::getCurrentCUDAStream().stream();'


def get_all_reverse_maps():
    """Merge all maps."""
    merged = {}
    merged.update(REVERSE_RUNTIME_API)
    merged.update(REVERSE_HEADER)
    merged.update(REVERSE_MACRO)
    return merged


# =============================================================================
# Replacement engine (acompute style)
# =============================================================================

def build_pattern(maps):
    """Build compiled regex from maps dict, longest key first."""
    sorted_keys = sorted(maps.keys(), key=lambda x: -len(x))
    return re.compile("|".join(re.escape(k) for k in sorted_keys))


def replace_content(content, maps, pattern):
    """Apply dict-based replacement."""
    return pattern.sub(lambda m: maps[m.group(0)], content)


def apply_reverse_arch_values(content):
    """Replace arch values only on lines containing __CUDA_ARCH__."""
    lines = content.split('\n')
    new_lines = []
    for line in lines:
        if '__CUDA_ARCH__' in line:
            for old, new in REVERSE_ARCH_VALUE.items():
                line = line.replace(old, new)
        new_lines.append(line)
    return '\n'.join(new_lines)


def apply_stream_cast_removal(content):
    """Remove (hggcStream_t) cast from stream assignments."""
    return STREAM_CAST_PATTERN.sub(STREAM_CAST_REPLACEMENT, content)


# =============================================================================
# Structural rewrites (pattern-based, no git)
# =============================================================================

def rewrite_flash_h():
    """Remove #ifdef __HGGCCC__ block from flash.h, replace with cuda includes.

    Handles both old format (hand-written PhiloxCudaState) and new format
    (direct hggc_runtime.h + ATen/cuda/CUDAGeneratorImpl.h).
    """
    fp = 'csrc/flash_attn/src/flash.h'
    if not os.path.isfile(fp):
        return False
    content = open(fp).read()
    if '#ifdef __HGGCCC__' not in content:
        return False

    # New format: #ifdef __HGGCCC__ / <hggc_runtime.h> / #else / typedef ... / #endif / <vector> / <ATen...>
    pat_new = re.compile(
        r'#ifdef __HGGCCC__\n'
        r'#include <hggc_runtime\.h>\n'
        r'#else\n'
        r'typedef struct HGstream_st\* hggcStream_t;\n'
        r'#endif\n'
        r'#include <vector>\n'
        r'#include <ATen/cuda/CUDAGeneratorImpl\.h>',
        re.DOTALL)
    replacement_new = (
        '#include <cuda.h>\n'
        '#include <vector>\n'
        '\n'
        '#include <ATen/cuda/CUDAGeneratorImpl.h> // For at::Generator and at::PhiloxCudaState'
    )
    new_content = pat_new.sub(replacement_new, content)

    if new_content == content:
        # Try old format: hand-written PhiloxCudaState
        pat_old = re.compile(
            r'#ifdef __HGGCCC__\n.*?#endif\s*\n\s*#include <vector>',
            re.DOTALL)
        replacement_old = (
            '#include <cuda.h>\n'
            '#include <vector>\n'
            '\n'
            '#include <ATen/cuda/CUDAGeneratorImpl.h> // For at::Generator and at::PhiloxCudaState'
        )
        new_content = pat_old.sub(replacement_old, content)

    if new_content == content:
        return False
    open(fp, 'w').write(new_content)
    return True


def rewrite_hardware_info_h():
    """Remove hggc API from hardware_info.h, replace with cuda API.

    Handles both old format (duplicate #ifdef __HGGCCC__ branches) and new format
    (simplified single-path with hggc API).
    """
    fp = 'csrc/flash_attn/src/hardware_info.h'
    if not os.path.isfile(fp):
        return False
    content = open(fp).read()
    if 'hggcGetDevice' not in content and 'hggcDeviceGetAttribute' not in content:
        return False

    # Replace entire file with clean cuda version
    new_content = '''/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cstdio>
#include <cstdlib>
#include <tuple>

#if !defined(__CUDACC_RTC__)
#include "cuda_runtime.h"
#endif

#define CHECK_CUDA(call)                                                       \\
  do {                                                                         \\
    cudaError_t status_ = call;                                                \\
    if (status_ != cudaSuccess) {                                              \\
      fprintf(stderr, "CUDA error (%s:%d): %s\\\\n", __FILE__, __LINE__,          \\
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
'''
    open(fp, 'w').write(new_content)
    return True


def rewrite_philox_unpack():
    """Remove #ifdef __HGGCCC__ dual-path from philox_unpack.cuh."""
    fp = 'csrc/flash_attn/src/philox_unpack.cuh'
    if not os.path.isfile(fp):
        return False
    content = open(fp).read()
    if '#ifdef __HGGCCC__' not in content:
        return False

    new_content = (
        '// This is purely so that it works with torch 2.1. '
        'For torch 2.2+ we can include ATen/cuda/PhiloxUtils.cuh\n'
        '#pragma once\n'
        '#include <ATen/cuda/detail/UnpackRaw.cuh>\n'
    )
    open(fp, 'w').write(new_content)
    return True


def rewrite_utils_h():
    """Remove USE_CLANG guarded block, restore original cuda fp16/bf16 includes."""
    fp = 'csrc/flash_attn/src/utils.h'
    if not os.path.isfile(fp):
        return False
    content = open(fp).read()
    if 'USE_CLANG' not in content:
        return False

    # Pattern: #if defined(USE_CLANG) block
    pat = re.compile(
        r'#if defined\(USE_CLANG\)\n'
        r'#include <hggc_fp16\.h>\n'
        r'#include <hggc_bf16\.h>\n'
        r'#else\n'
        r'#include <hggc_fp16\.h>\n'
        r'#if defined\(__HGGC_ARCH__\) && __HGGC_ARCH__ >= 100\n'
        r'#include <hggc_bf16\.h>\n'
        r'#endif\n'
        r'#endif',
        re.DOTALL)

    replacement = (
        '#include <cuda_fp16.h>\n'
        '\n'
        '#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800\n'
        '#include <cuda_bf16.h>\n'
        '#endif'
    )

    new_content = pat.sub(replacement, content)
    if new_content == content:
        return False
    open(fp, 'w').write(new_content)
    return True


def rewrite_launch_template_guards():
    """Remove #ifndef __HGGCCC__ guards, restore original c10/ATen includes."""
    replacements = [
        ('csrc/flash_attn/src/flash_fwd_launch_template.h',
         '#ifndef __HGGCCC__\n#include <c10/cuda/CUDAException.h>\n#include <ATen/cuda/CUDAContext.h>\n#else\n#define C10_CUDA_CHECK(x) (void)(x)\n#define C10_CUDA_KERNEL_LAUNCH_CHECK()\n#endif',
         '#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK\n#include <ATen/cuda/CUDAContext.h>'),
        ('csrc/flash_attn/src/flash_bwd_launch_template.h',
         '#ifndef __HGGCCC__\n#include <c10/cuda/CUDAException.h>\n#else\n#define C10_CUDA_CHECK(x) (void)(x)\n#define C10_CUDA_KERNEL_LAUNCH_CHECK()\n#endif',
         '#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK'),
    ]
    count = 0
    for fp, old, new in replacements:
        if not os.path.isfile(fp):
            continue
        content = open(fp).read()
        if old in content:
            open(fp, 'w').write(content.replace(old, new))
            count += 1
    return count


def delete_hggcrt_driver_types_shim():
    """Delete the hggcrt_driver_types.h shim file (not needed in cuda-compat mode)."""
    targets = [
        'csrc/flash_attn/src/hggcrt_driver_types.h',
        'hopper/hggcrt_driver_types.h',
    ]
    count = 0
    for fp in targets:
        if os.path.isfile(fp):
            os.remove(fp)
            count += 1
    return count


def _fa3_ppu_restore(new_content):
    """Restore PPU-specific build structures for FA3 hopper/setup.py after transformation.

    The ppu-original setup.py drops the USE_PPU conditional structure, DISABLE_SM80/SM89
    variables, and the _write_ninja_file monkey patch that strips PTX code (code=compute_*)
    from the ELF. Without that patch, a full PPU build (107 sources) hits a PC-relative
    offset overflow at link time. This restores parity with the original cuda-compatible
    base (16b4b34) for every PPU-relevant path, while leaving non-PPU logic untouched.
    """
    # 1. Add USE_PPU / FORCE_BUILD / SKIP_CUDA_BUILD right after this_dir
    new_content = new_content.replace(
        'PACKAGE_NAME = "flash_attn_3"\n'
        'this_dir = os.path.dirname(os.path.abspath(__file__))\n'
        '\n'
        'SKIP_KERNEL_BUILD',
        'PACKAGE_NAME = "flash_attn_3"\n'
        'this_dir = os.path.dirname(os.path.abspath(__file__))\n'
        '\n'
        "USE_PPU = 'PPU_SDK' in os.environ.keys()\n"
        'FORCE_BUILD = os.getenv("FLASH_ATTENTION_FORCE_BUILD", ("TRUE" if USE_PPU else "FALSE")) == "TRUE"\n'
        'SKIP_CUDA_BUILD = os.getenv("FLASH_ATTENTION_SKIP_CUDA_BUILD", "FALSE") == "TRUE"\n'
        '\n'
        'SKIP_KERNEL_BUILD',
        1)

    # 2. DISABLE_CLUSTER default must depend on USE_PPU, matching the original base
    new_content = new_content.replace(
        'DISABLE_CLUSTER = os.getenv("FLASH_ATTENTION_DISABLE_CLUSTER", "TRUE") == "TRUE"',
        'DISABLE_CLUSTER = os.getenv("FLASH_ATTENTION_DISABLE_CLUSTER", ("TRUE" if USE_PPU else "FALSE")) == "TRUE"',
        1)

    # 3. DISABLE_SM8x env-var key must depend on USE_PPU
    new_content = new_content.replace(
        'DISABLE_SM8x = os.getenv("FLASH_ATTENTION_DISABLE_SM8x", "FALSE") == "TRUE"',
        'DISABLE_SM8x = os.getenv("FLASH_ATTENTION_DISABLE_SM8x" if USE_PPU else "FLASH_ATTENTION_DISABLE_SM80", "FALSE") == "TRUE"',
        1)

    # 4. Replace hardcoded DISABLE_SM90=True with the original env-based form,
    #    and restore DISABLE_SM80 / DISABLE_SM89 (used by the ninja monkey patch below)
    new_content = new_content.replace(
        'DISABLE_SM90 = True  # SM90 removed from this PPU-only build',
        'DISABLE_SM90 = os.getenv("FLASH_ATTENTION_DISABLE_SM90", ("TRUE" if USE_PPU else "FALSE")) == "TRUE"\n'
        'DISABLE_SM80 = os.getenv("FLASH_ATTENTION_DISABLE_SM80", "FALSE") == "TRUE"\n'
        'DISABLE_SM89 = os.getenv("FLASH_ATTENTION_DISABLE_SM89", "FALSE") == "TRUE"',
        1)

    # 5. Restore original guard name (SKIP_CUDA_BUILD instead of SKIP_KERNEL_BUILD)
    new_content = new_content.replace('if not SKIP_KERNEL_BUILD:', 'if not SKIP_CUDA_BUILD:', 1)

    # 6. Inject the PPU _write_ninja_file monkey patch before ext_modules = [].
    #    This is the part that removes code=compute_* (PTX) from the ELF, which is
    #    what the original cuda-compatible setup.py does to avoid link-time PLT/
    #    PC-relative offset overflow on large PPU builds.
    patch = (
        '# PPU: monkey-patch ninja file writer to strip PTX code (code=compute_*) from the\n'
        '# ELF, preventing PC-relative offset overflow at link time on large PPU builds.\n'
        'if USE_PPU and hasattr(torch.utils.cpp_extension, "_write_ninja_file"):\n'
        '    _orig_write_ninja_file = torch.utils.cpp_extension._write_ninja_file\n'
        '\n'
        '    def _ppu_write_ninja_file(path, cflags=None, post_cflags=None,\n'
        '                              cuda_cflags=None, cuda_post_cflags=None,\n'
        '                              cuda_dlink_post_cflags=None, sources=None,\n'
        '                              objects=None, ldflags=None, library_target=None,\n'
        '                              with_cuda=None, **kwargs):\n'
        '        """Replace code=compute_* with code=sm_* in all nvcc flags for PPU."""\n'
        '        def _fix(flags):\n'
        '            if not flags:\n'
        '                return flags\n'
        '            r = [s.replace("code=compute_", "code=sm_") for s in flags]\n'
        '            if DISABLE_SM80:\n'
        '                r = [s.replace("sm_80", "sm_89") for s in r]\n'
        '            if DISABLE_SM89:\n'
        '                r = [s.replace("sm_80", "sm_80a") for s in r]\n'
        '            return r\n'
        '        return _orig_write_ninja_file(\n'
        '            path, cflags=_fix(cflags), post_cflags=_fix(post_cflags),\n'
        '            cuda_cflags=_fix(cuda_cflags), cuda_post_cflags=_fix(cuda_post_cflags),\n'
        '            cuda_dlink_post_cflags=cuda_dlink_post_cflags,\n'
        '            sources=sources, objects=objects, ldflags=ldflags,\n'
        '            library_target=library_target, with_cuda=with_cuda, **kwargs)\n'
        '\n'
        '    torch.utils.cpp_extension._write_ninja_file = _ppu_write_ninja_file\n'
        '\n'
        '\n'
    )
    insert_pos = new_content.find('ext_modules = []')
    if insert_pos > 0:
        new_content = new_content[:insert_pos] + patch + new_content[insert_pos:]

    # 7. Restore original setup() metadata: wheel ABI options and python_requires
    new_content = new_content.replace('    python_requires=">=3.9",', '    python_requires=">=3.8",', 1)
    new_content = new_content.replace(
        '    install_requires=["torch", "einops"],\n'
        '    setup_requires=["packaging", "psutil", "ninja"],\n'
        ')',
        '    install_requires=["torch", "einops", "packaging", "ninja"],\n'
        '    options={"bdist_wheel": {"py_limited_api": "cp39"}},\n'
        ')',
        1)

    return new_content


def rewrite_setup_py():
    """Transform setup.py from HGCCBuildExtension to CUDAExtension/BuildExtension.

    Handles new format: extra_compile_args={"hgcc":..., "cxx":...}
    """
    targets = ['setup.py', 'hopper/setup.py']
    count = 0
    for fp in targets:
        if not os.path.isfile(fp):
            continue
        content = open(fp).read()
        if 'HGCCBuildExtension' not in content:
            continue

        new_content = content

        # --- Step 1: Fix imports ---
        if 'from torch.utils.cpp_extension import' not in new_content:
            new_content = new_content.replace(
                'from setuptools import setup, find_packages, Extension\n'
                'from setuptools.command.build_ext import build_ext',
                'from setuptools import setup, find_packages\n'
                'from setuptools.command.build_ext import build_ext\n'
                'from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME')

        # --- Step 2: Remove HGCCBuildExtension class ---
        # Match from comment block to 3+ newlines (2+ blank lines = class boundary)
        # This preserves get_package_version() and CachedWheelsCommand() that follow
        hgcc_class_pat = re.compile(
            r'# =+\n# PPU HGCC Build Extension.*?\n{3,}',
            re.DOTALL)
        new_content = hgcc_class_pat.sub('', new_content)

        # --- Step 3: Replace Extension() with CUDAExtension() ---
        new_content = re.sub(
            r'\bExtension\(',
            'CUDAExtension(',
            new_content)

        # --- Step 4: Replace cmdclass ---
        new_content = new_content.replace(
            '"build_ext": HGCCBuildExtension',
            '"build_ext": BuildExtension')

        # --- Step 5: Remove -arch flags ---
        new_content = new_content.replace('        "-arch=ppu_10",\n', '')
        new_content = new_content.replace('        "-arch=ppu_15",\n', '')
        new_content = new_content.replace('            "-arch=ppu_10",\n', '')
        new_content = new_content.replace('            "-arch=ppu_15",\n', '')

        # --- Step 6: Remove ppu-original-only macros ---
        new_content = new_content.replace('        "-DSWITCH_TO_HGGCRT",\n', '')
        new_content = new_content.replace('            "-DSWITCH_TO_HGGCRT",\n', '')
        new_content = new_content.replace('        "-DUSE_CLANG", "-DUSE_HGGC", "-DUSE_PPU", "-DUSE_AIU=1",\n',
                                          '        "-DUSE_PPU", "-DUSE_AIU=1",\n')
        new_content = new_content.replace('            "-DUSE_CLANG", "-DUSE_HGGC", "-DUSE_PPU", "-DUSE_AIU=1",\n',
                                          '            "-DUSE_PPU", "-DUSE_AIU=1",\n')
        new_content = new_content.replace('        "-DUSE_CLANG",\n', '')
        new_content = new_content.replace('        "-DUSE_HGGC",\n', '')
        new_content = new_content.replace('            "-DUSE_CLANG",\n', '')
        new_content = new_content.replace('            "-DUSE_HGGC",\n', '')

        # --- Step 6b: Replace extra_compile_args keys and HGGC macros ---
        new_content = new_content.replace('"hgcc"', '"nvcc"')
        new_content = new_content.replace('"hgcc_extra"', '"nvcc"')
        new_content = new_content.replace('"cxx_extra"', '"cxx"')
        # HGGC macros → CUDA macros
        new_content = new_content.replace('__HGGC_NO_HALF_OPERATORS__', '__CUDA_NO_HALF_OPERATORS__')
        new_content = new_content.replace('__HGGC_NO_HALF_CONVERSIONS__', '__CUDA_NO_HALF_CONVERSIONS__')
        new_content = new_content.replace('__HGGC_NO_HALF2_OPERATORS__', '__CUDA_NO_HALF2_OPERATORS__')
        new_content = new_content.replace('__HGGC_NO_BFLOAT16_CONVERSIONS__', '__CUDA_NO_BFLOAT16_CONVERSIONS__')

        # --- Step 6c: Align FA2 cuda-compatible flag order ---
        # FA2 ppu-original setup.py already has hgcc_flags; keep its final order aligned
        # with the original ppu-original nvcc_flags.
        if 'PACKAGE_NAME = "flash_attn_3"' not in new_content:
            new_content = new_content.replace(
                'cxx_flags = ["-O3", "-std=c++17", "-fPIC", "-DUSE_PPU", "-DUSE_AIU=1"]',
                'cxx_flags = ["-O3", "-std=c++17"]')
            fa2_early_ppu_flags = (
                '        "--use_fast_math",\n'
                '        "-DUSE_PPU",\n'
                '        "-DUSE_AIU=1",\n'
                '        "-mllvm", "-ppu-max-vreg-count=256",'
            )
            if fa2_early_ppu_flags in new_content:
                new_content = new_content.replace(
                    fa2_early_ppu_flags,
                    '        "--use_fast_math",\n'
                    '        "-mllvm", "-ppu-max-vreg-count=256",',
                    1)
                new_content = new_content.replace(
                    '        "-mllvm", "-ppu-alloca-half-ldst-simplify=true",\n'
                    '    ]',
                    '        "-mllvm", "-ppu-alloca-half-ldst-simplify=true",\n'
                    '        "-DUSE_PPU",\n'
                    '        "-DUSE_AIU=1",\n'
                    '    ]',
                    1)

        # --- Step 7: Remove ppu-original trailing block ---
        cuda_free_block = re.compile(
            r'\n# CUDA-Free mode:.*$', re.DOTALL)
        new_content = cuda_free_block.sub('\n', new_content)

        # --- Step 8: Add hgcc_flags if CUDAExtension used ---
        # FA2: no extra_compile_args → insert new block
        # FA3: has extra_compile_args with feature_args → add hgcc_flags definition
        if 'hgcc_flags' not in new_content and 'CUDAExtension(' in new_content:
            if 'PACKAGE_NAME = "flash_attn_3"' in new_content:
                nvcc_block = '''    hgcc_flags = [
        "-O3", "-std=c++17",
        "--ftemplate-backtrace-limit=0",
        "--use_fast_math",
        "--resource-usage",
        "-lineinfo",
        "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",
        "-DCUTLASS_ENABLE_GDC_FOR_SM90",
        "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
        "-DNDEBUG",
        "-mllvm", "-ppu-max-vreg-count=256",
        "-mllvm", "-ppu-sink-matrix-addr=true",
        "-mllvm", "-ppu-max-alloca-byte-size=320",
        "-mllvm", "-ppu-sink-async-addr=true",
        "-mllvm", "-ppu-sink-load-addr=true",
        "-mllvm", "-ppu-sink-store-addr=true",
        "-mllvm", "-ppu-alloca-half-ldst-simplify=true",
        "-mllvm", "-ppu-volatile-yield=false",
        "-mllvm", "-sort-copy-before-coalesce=true",
        "-Xfatbin",
        "--compress-all",
    ]\n\n'''
            else:
                nvcc_block = '''    hgcc_flags = [
        "-O3", "-std=c++17",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "-mllvm", "-ppu-max-vreg-count=256",
        "-mllvm", "-ppu-sink-matrix-addr=true",
        "-mllvm", "-ppu-max-alloca-byte-size=320",
        "-mllvm", "-ppu-sink-async-addr=true",
        "-mllvm", "-ppu-sink-load-addr=true",
        "-mllvm", "-ppu-sink-store-addr=true",
        "-mllvm", "-ppu-alloca-half-ldst-simplify=true",
        "-DUSE_PPU", "-DUSE_AIU=1",
    ]\n\n'''
            insert_pos = new_content.find('    ext_modules.append(')
            if insert_pos == -1:
                insert_pos = new_content.find('ext_modules.append(')
            if insert_pos > 0:
                new_content = new_content[:insert_pos] + nvcc_block + new_content[insert_pos:]

            # FA2: no extra_compile_args → insert new block
            if 'extra_compile_args' not in new_content:
                new_content = new_content.replace(
                    '        )\n    )\n',
                    '            extra_compile_args={"cxx": ["-O3", "-std=c++17"], "nvcc": hgcc_flags},\n'
                    '        )\n    )\n', 1)

        # --- Step 8b: FA3 cxx args add -O3 -std=c++17 ---
        if 'CUDAExtension(' in new_content and '"-DPy_LIMITED_API=' in new_content:
            new_content = new_content.replace(
                '"cxx": ["-DPy_LIMITED_API=',
                '"cxx": ["-O3", "-std=c++17", "-DPy_LIMITED_API=')

        # --- Step 9: Add cc_flag for gencode arch ---
        # PPU GPU is not standard CUDA, so PyTorch's auto-detection fails.
        # Add explicit -gencode flags and append to nvcc args.
        # Note: check 'cc_flag =' not 'cc_flag' because 'cc_flag' is substring of 'hgcc_flags'
        if 'cc_flag =' not in new_content and 'CUDAExtension(' in new_content:
            if 'PACKAGE_NAME = "flash_attn_3"' in new_content:
                # FA3: match original cuda-compatible logic
                # cc_flag = []
                # if not DISABLE_SM90: sm_90a
                # PPU default: DISABLE_SM90=True → cc_flag stays empty
                gencode_block = '''    cc_flag = []
    if not DISABLE_SM90:
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_90a,code=sm_90a")

'''
            else:
                # FA2: PPU default cuda_archs() returns ["80, 89"]
                gencode_block = '''    cc_flag = []
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_80,code=sm_80")
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_89,code=sm_89")

'''
            insert_pos = new_content.find('    ext_modules.append(')
            if insert_pos == -1:
                insert_pos = new_content.find('ext_modules.append(')
            if insert_pos > 0:
                new_content = new_content[:insert_pos] + gencode_block + new_content[insert_pos:]

            # Append cc_flag to nvcc args
            for old, new in [
                ('"nvcc": hgcc_flags,', '"nvcc": hgcc_flags + cc_flag,'),
                ('"nvcc": nvcc_flags,', '"nvcc": nvcc_flags + cc_flag,'),
                ('"nvcc": hgcc_flags\n', '"nvcc": hgcc_flags + cc_flag\n'),
                ('"nvcc": nvcc_flags\n', '"nvcc": nvcc_flags + cc_flag\n'),
                # FA3 pattern: feature_args (no hgcc_flags/nvcc_flags variable)
                ('"nvcc": feature_args,', '"nvcc": hgcc_flags + cc_flag + feature_args,'),
                ('"nvcc": feature_args\n', '"nvcc": hgcc_flags + cc_flag + feature_args\n'),
            ]:
                new_content = new_content.replace(old, new)

            # Handle inline nvcc flags (not using a variable)
            new_content = re.sub(
                r'("nvcc":\s*\[.*?"-DUSE_PPU",\s*"-DUSE_AIU=1")\]',
                r'\1] + cc_flag',
                new_content, flags=re.DOTALL)

        # --- Step 10: Add append_nvcc_threads helper ---
        # Original setup.py wraps nvcc args with append_nvcc_threads() for --threads flag
        if 'append_nvcc_threads' not in new_content and 'CUDAExtension(' in new_content:
            helper = '''def append_nvcc_threads(nvcc_extra_args):
    nvcc_threads = os.getenv("NVCC_THREADS") or "2"
    return nvcc_extra_args + ["--threads", nvcc_threads]

'''
            # Insert BEFORE 'if not SKIP_KERNEL_BUILD' to avoid breaking the if block
            insert_pos = new_content.find('if not SKIP_KERNEL_BUILD:')
            if insert_pos == -1:
                insert_pos = new_content.find('if not SKIP_CUDA_BUILD:')
            if insert_pos > 0:
                new_content = new_content[:insert_pos] + helper + new_content[insert_pos:]

            # Wrap nvcc args with append_nvcc_threads()
            for pattern in [
                '"nvcc": hgcc_flags + cc_flag',
                '"nvcc": nvcc_flags + cc_flag',
                '"nvcc": hgcc_flags + cc_flag + feature_args',
            ]:
                new_content = new_content.replace(
                    pattern,
                    pattern.replace('"nvcc": ', '"nvcc": append_nvcc_threads(') + ')')

        # --- Step 10b: FA3 keeps original cuda-compatible --threads ordering ---
        if 'PACKAGE_NAME = "flash_attn_3"' in new_content and 'append_nvcc_threads' in new_content:
            new_content = new_content.replace(
                'def append_nvcc_threads(nvcc_extra_args):\n'
                '    nvcc_threads = os.getenv("NVCC_THREADS") or "2"\n'
                '    return nvcc_extra_args + ["--threads", nvcc_threads]\n\n',
                'def nvcc_threads_args():\n'
                '    nvcc_threads = os.getenv("NVCC_THREADS") or "2"\n'
                '    return ["--threads", nvcc_threads]\n\n',
                1)
            new_content = new_content.replace(
                '"nvcc": append_nvcc_threads(hgcc_flags + cc_flag) + feature_args,',
                '"nvcc": nvcc_threads_args() + hgcc_flags + cc_flag + feature_args,',
                1)

        # --- Step 11: Add py_limited_api=True for FA3 (prevents PLT overflow) ---
        if 'py_limited_api' not in new_content and 'CUDAExtension(' in new_content and '"-DPy_LIMITED_API=' in new_content:
            # FA3 pattern: },\n        )\n    )  (no comma after closing paren)
            new_content = new_content.replace(
                '            },\n        )\n    )',
                '            },\n            py_limited_api=True,\n        )\n    )')
            # FA2 pattern: },\n        ),  (with comma)
            new_content = new_content.replace(
                '            },\n        ),',
                '            },\n            py_limited_api=True,\n        ),')

        # --- Step 12: FA3 PPU structure restore ---
        # Restores USE_PPU, DISABLE_SM80/SM89, _write_ninja_file monkey patch,
        # and setup() metadata to match the original cuda-compatible base (16b4b34).
        if 'PACKAGE_NAME = "flash_attn_3"' in new_content:
            new_content = _fa3_ppu_restore(new_content)

        if new_content != content:
            open(fp, 'w').write(new_content)
            count += 1
    return count


# =============================================================================
# ACTLIZE CUDA-compat shim (SM75/SM80 aliases, thin wrappers, arch macro map)
#
# The shared actlize converter only renames PPU/HGGC symbols; it does not emit
# the SM75_/SM80_ aliases that flash-attention kernels reference (Copy_Atom<
# SM75_U32x4_LDSM_N>, SM80_CP_ASYNC_CACHEGLOBAL, ...).  These generators were
# removed from the actlize converter in 9cb2a26; without them the kernel .cu
# TUs fail with SM75_/SM80_ undeclared.  They are restored here (verbatim from
# 9cb2a26~1) and applied to the actlize cutlass include tree, scoped to flex so
# the shared actlize converter and other consumers stay untouched.
# =============================================================================

# Base PPU/HGGC -> CUDA naming applied to the actlize include tree.  This is a
# self-contained copy of the shared actlize converter's NAMING_REPLACEMENTS so
# the flex flow can cudafy its bundled actlize checkout on its own, without the
# cudafy dispatcher routing through the shared _convert_actlize step.  Order
# matters: longer patterns before shorter ones.  __HGGC_ARCH__ is intentionally
# NOT mapped here (the arch macro map header handles 100/150 <-> 800/890).
ACTLIZE_BASE_NAMING = [
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
    (r'\bhggcDevAttrMultiProcessorCount\b', 'cudaDevAttrMultiProcessorCount'),
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

    # === Category 5: System header includes (hggc -> cuda; omit <> to handle both quote styles) ===
    (r'\bhggc_runtime_api\.h\b', 'cuda_runtime_api.h'),
    (r'\bhggc_runtime\.h\b', 'cuda_runtime.h'),
    (r'\bhggc_fp16\.h\b', 'cuda_fp16.h'),
    (r'\bhggc_fp8\.h\b', 'cuda_fp8.h'),
    (r'\bhggc_bf16\.h\b', 'cuda_bf16.h'),
    (r'\bacComplex\.h\b', 'cuComplex.h'),
    (r'\bhgComplex\.h\b', 'cuComplex.h'),
    (r'\bacrand_kernel\.h\b', 'curand_kernel.h'),

    # === ac* complex type -> cu* mapping (long patterns first) ===
    (r'\bmake_acDoubleComplex\b', 'make_cuDoubleComplex'),
    (r'\bmake_acFloatComplex\b', 'make_cuFloatComplex'),
    (r'\bacDoubleComplex\b', 'cuDoubleComplex'),
    (r'\bacFloatComplex\b', 'cuFloatComplex'),
    (r'\bacCrealf\b', 'cuCrealf'),
    (r'\bacCimagf\b', 'cuCimagf'),
    (r'\bacCreal\b', 'cuCreal'),
    (r'\bacCimag\b', 'cuCimag'),
    (r'\bis_acComplex\b', 'is_cuComplex'),
    (r'\bacComplex\.h\b', 'cuComplex.h'),
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
    (r'\bacrand\b', 'curand'),

    # === CUTLASS internal macros + compiler macro mapping ===
    (r'\bHGlaunchAttribute\b', 'CUlaunchAttribute'),
    (r'\bhgTensorMapEncodeIm2col\b', 'cuTensorMapEncodeIm2col'),

    # Platform macros
    (r'\b__HGGC_STD_MAX\b', '__NV_STD_MAX'),
    (r'\b__HGGC_STD_MIN\b', '__NV_STD_MIN'),

    # CUDA API constants
    (r'\bHGGC_SUCCESS\b', 'CUDA_SUCCESS'),
    (r'\bHGGC_ERROR_UNKNOWN\b', 'CUDA_ERROR_UNKNOWN'),
    (r'\bhggcErrorUnknown\b', 'cudaErrorUnknown'),

    # Compiler/runtime version macros (HGGC -> CUDA, longer patterns first)
    (r'\bHGGCRT_VERSION\b', 'CUDART_VERSION'),
    (r'\b__HGGCCC_RTC__\b', '__CUDACC_RTC__'),
    (r'\b__HGGCCC_VER_MAJOR__\b', '__CUDACC_VER_MAJOR__'),
    (r'\b__HGGCCC_VER_MINOR__\b', '__CUDACC_VER_MINOR__'),
    (r'\b__HGGCCC_VERSION__\b', '__CUDACC_VERSION__'),
    (r'\b__HGGCCC__\b', '__CUDACC__'),

    # Clang CUDA macro
    (r'\b__HGGC__\b', '__CUDA__'),

    # === namespace (hggc:: -> cuda::) ===
    (r'\bhggc::', 'cuda::'),
]


# Cu/Ptg -> Sm/Nv identifier renames in the actlize headers so upstream-style
# cutlass names resolve (KernelHardwareInfo::cu_count -> sm_count, etc.).  These
# rules lived in the actlize converter's NAMING_REPLACEMENTS until 9cb2a26
# dropped them; the flash-attention kernels reference the upstream names, so
# without this the TUs fail with "no member named 'sm_count'".  Restored here
# (verbatim from 9cb2a26~1) scoped to flex.
ACTLIZE_CU_SM_RENAMES = [
    (r'\bcu_count\b', 'sm_count'),
    (r'\bdevice_cus_\b', 'device_sms_'),
    (r'\bcu_occupancy_\b', 'sm_occupancy_'),
    (r'\bdevice_cus\b', 'device_sms'),
    (r'\bcu_occupancy\b', 'sm_occupancy'),
    (r'\bavail_cus_\b', 'avail_sms_'),
    (r'\bavail_cus\b', 'avail_sms'),
    (r'\bavailable_cu_count\b', 'available_sm_count'),
    (r'\boverride_cu_count\b', 'override_sm_count'),
    (r'\bmax_cu_per_gpc\b', 'max_sm_per_gpc'),
    (r'\btb_per_cu\b', 'tb_per_sm'),
    (r'\bMaxCuCount\b', 'MaxSmCount'),
    (r'\bCuArch\b', 'SmArch'),
    (r'\bdest_cu\b', 'dest_sm'),
    (r'\bPtgType\b', 'NvType'),
    (r'\bPtgTypeV2\b', 'NvTypeV2'),
]


def apply_actlize_naming_fixups(include_dir, changed):
    """Apply the base PPU/HGGC -> CUDA naming plus the Cu/Ptg -> Sm/Nv renames to
    the actlize cutlass headers (and the sibling tools/util/include tree, matching
    the actlize converter's walk).  This is the full self-contained actlize name
    conversion: ACTLIZE_BASE_NAMING mirrors the shared actlize converter, and
    ACTLIZE_CU_SM_RENAMES restores the upstream cutlass member names.
    Idempotent: word-boundary renames leave nothing to match on a second pass."""
    compiled = [(re.compile(p), r) for p, r in ACTLIZE_BASE_NAMING + ACTLIZE_CU_SM_RENAMES]
    code_exts = ('.h', '.hpp', '.cuh', '.cu', '.c', '.cpp', '.inl', '.inc')
    roots = [include_dir]
    tools_util = os.path.normpath(
        os.path.join(include_dir, '..', 'tools', 'util', 'include'))
    if os.path.isdir(tools_util):
        roots.append(tools_util)
    for base in roots:
        for root, dirs, files in os.walk(base):
            for fname in files:
                if not fname.endswith(code_exts):
                    continue
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                except Exception:
                    continue
                new_content = content
                for pat, repl in compiled:
                    new_content = pat.sub(repl, new_content)
                if new_content != content:
                    with open(fpath, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    changed.append(
                        f"{os.path.relpath(fpath, include_dir)} (actlize naming)")


PPU_ALIAS_INJECTIONS = {
    "cutlass/ppu_host_adapter.hpp": ("cutlass", """\
// Host adapter type aliases
using CudaHostLaunchAttributes = HostLaunchAttributes;
using CudaHostAdapter = HostAdapter;
"""),
    "cutlass/arch/arch.h": ("cutlass::arch", """\
// SM architecture tag aliases
using Sm50 = cutlass::arch::PPU0010;
using Sm60 = cutlass::arch::PPU0010;
using Sm61 = cutlass::arch::PPU0010;
using Sm70 = cutlass::arch::PPU0010;
using Sm72 = cutlass::arch::PPU0010;
using Sm75 = cutlass::arch::PPU0010;
using Sm80 = cutlass::arch::PPU0010;
using Sm86 = cutlass::arch::PPU0010;
using Sm89 = cutlass::arch::PPU0015;
using Sm90 = cutlass::arch::PPU0015;

// SmId function alias
CUTLASS_DEVICE
int SmId() { return cutlass::arch::CuId(); }
"""),
    "cute/arch/copy_ppu.hpp": ("cute", """\
// SM75 ldmatrix macro aliases
#ifndef CUTE_ARCH_LDSM_SM75_ENABLED
#define CUTE_ARCH_LDSM_SM75_ENABLED CUTE_ARCH_LDSM_PPU_ENABLED
#endif
#ifndef CUTE_ARCH_LDSM_SM75_ACTIVATED
#define CUTE_ARCH_LDSM_SM75_ACTIVATED CUTE_ARCH_LDSM_PPU_ACTIVATED
#endif
// SM80 cp.async macro alias
#ifndef CUTE_ARCH_CP_ASYNC_SM80_ENABLED
#define CUTE_ARCH_CP_ASYNC_SM80_ENABLED CUTE_ARCH_CP_ASYNC_PPU_ENABLED
#endif

// SM75 ldmatrix
using SM75_U32x1_LDSM_N = PPU_U32x1_LDSM_N;
using SM75_U32x2_LDSM_N = PPU_U32x2_LDSM_N;
using SM75_U32x4_LDSM_N = PPU_U32x4_LDSM_N;
using SM75_U16x2_LDSM_T = PPU_U16x2_LDSM_T;
using SM75_U16x4_LDSM_T = PPU_U16x4_LDSM_T;
using SM75_U16x8_LDSM_T = PPU_U16x8_LDSM_T;
// SM80 cp.async (class templates)
template <class TS, class TD = TS> using SM80_CP_ASYNC_CACHEALWAYS = PPU_CP_ASYNC_CACHEALWAYS<TS, TD>;
template <class TS, class TD = TS> using SM80_CP_ASYNC_CACHEGLOBAL = PPU_CP_ASYNC_CACHEGLOBAL<TS, TD>;
template <class TS, class TD = TS> using SM80_CP_ASYNC_CACHEALWAYS_ZFILL = PPU_CP_ASYNC_CACHEALWAYS_ZFILL<TS, TD>;
template <class TS, class TD = TS> using SM80_CP_ASYNC_CACHEGLOBAL_ZFILL = PPU_CP_ASYNC_CACHEGLOBAL_ZFILL<TS, TD>;
"""),
    "cute/arch/mma_ppu.hpp": ("cute", """\
// SM61 DP4A
using SM61_DP4A = PPU_DP4A;
"""),
    "cutlass/epilogue/dispatch_policy.hpp": ("cutlass::epilogue", """\
template <int StagesC_, int StagesD_, int FragmentSize_, bool ReuseSmemC_, bool DelayTmaStore_>
using Sm90TmaWarpSpecialized = PPUTmaWarpSpecialized<StagesC_, StagesD_, FragmentSize_, ReuseSmemC_, DelayTmaStore_>;
template <int StagesC_, int StagesD_, int FragmentSize_, bool ReuseSmemC_, bool DelayTmaStore_, int NumEpilogueWarpGroups_>
using Sm90PtrArrayTmaWarpSpecialized = PPUPtrArrayTmaWarpSpecialized<StagesC_, StagesD_, FragmentSize_, ReuseSmemC_, DelayTmaStore_, NumEpilogueWarpGroups_>;
template <int StagesC_, int StagesD_, int FragmentSize_ = 2>
using Sm90TmaWarpSpecializedBiasElementwise = PPUTmaWarpSpecializedBiasElementwise<StagesC_, StagesD_, FragmentSize_>;
"""),
    "cutlass/epilogue/collective/detail.hpp": ("cutlass::epilogue::collective::detail", """\
template<class Schedule>
static constexpr bool sm90_is_ptr_array_tma_v = ppu_is_ptr_array_tma_v<Schedule>;
"""),
    "cutlass/epilogue/fusion/ppu_visitor_tma_warpspecialized.hpp": ("cutlass::epilogue::fusion::detail", """\
template <class... Ops>
using Sm90VisitorImplBase = PPUVisitorImplBase<Ops...>;
"""),
    "cutlass/epilogue/threadblock/fusion/visitor_2x.hpp": ("cutlass::epilogue::threadblock", """\
template <class NodeOp, class... ChildOps>
using Sm80EVT = PPUEVT2x<NodeOp, ChildOps...>;
template <class ElementCompute, class EdgeTuple, class... Ops>
using Sm80TopologicalVisitor = PPUTopologicalVisitor<ElementCompute, EdgeTuple, Ops...>;
"""),
    "cutlass/gemm/dispatch_policy.hpp": ("cutlass::gemm", """\
using MainloopSm70TwoStageUnpredicated = MainloopPPUTwoStageUnpredicated;
using MainloopSm70TwoStage = MainloopPPUTwoStage;
template<int Stages_>
using MainloopSm80CpAsyncUnpredicated = MainloopPPUCpAsyncUnpredicated<Stages_>;
template<int Stages_, class ClusterShape_ = Shape<_1,_1,_1>>
using MainloopSm80CpAsync = MainloopPPUCpAsyncLegacy<Stages_, ClusterShape_>;
template<int Stages_>
using MainloopSm80CpAsyncWithScale = MainloopPPUCpAsyncWithScale<Stages_>;
template<int Stages_, typename Schedule_ = KernelMultistage>
using MainloopSm80CpAsyncBatchArray = MainloopPPUCpAsyncBatchArray<Stages_, Schedule_>;
using MainloopSm70TwoStageUnpredicatedLdmatrix = MainloopPPUTwoStageUnpredicatedLdmatrix;
using MainloopSm70TwoStageLdmatrix = MainloopPPUTwoStageLdmatrix;
using MainloopSm70TwoStageLdmatrixBatchArray = MainloopPPUTwoStageLdmatrixBatchArray;
"""),
    "cutlass/gemm/kernel/tile_scheduler_params.h": ("cutlass::gemm::kernel::detail", """\
using PersistentTileSchedulerSm90Params = PersistentTileSchedulerPPUParams;
using PersistentTileSchedulerSm90StreamKParams = PersistentTileSchedulerPPUStreamKParams;
template<class ProblemShape>
using PersistentTileSchedulerSm90GroupParams = PersistentTileSchedulerPPUGroupParams<ProblemShape>;
"""),
    "cutlass/gemm/kernel/tile_scheduler.hpp": ("cutlass::gemm::kernel::detail", """\
using PersistentTileSchedulerSm90 = PersistentTileSchedulerPPU;
template <class TileShape, class ClusterShape>
using PersistentTileSchedulerSm90StreamK = PersistentTileSchedulerPPUStreamK<TileShape, ClusterShape>;
template <class GroupProblemShape>
using PersistentTileSchedulerSm90Group = PersistentTileSchedulerPPUGroup<GroupProblemShape>;
"""),
}

# Thin wrapper file specs: (relative path, [headers to include])
# Wrapper files only redirect headers; aliases are provided transitively by PPU source files
WRAPPER_SPECS = [
    # --- cutlass layer: host adapter + nvrtc ---
    ("cutlass/cuda_host_adapter.hpp",     ['"cutlass/ppu_host_adapter.hpp"']),
    ("cutlass/floating_point_nvrtc.h",    ['"cutlass/floating_point_hgrtc.h"']),
    # --- cute/arch layer ---
    ("cute/arch/copy_sm75.hpp",          ["<cute/arch/copy_ppu.hpp>"]),
    ("cute/arch/copy_sm80.hpp",          ["<cute/arch/copy_ppu.hpp>"]),
    ("cute/arch/mma_sm61.hpp",           ["<cute/arch/mma_ppu.hpp>"]),
    # --- cute/atom layer ---
    ("cute/atom/copy_traits_sm75.hpp",   ["<cute/arch/copy_sm75.hpp>", "<cute/atom/copy_traits_ppu.hpp>"]),
    ("cute/atom/copy_traits_sm80.hpp",   ["<cute/arch/copy_sm80.hpp>", "<cute/atom/copy_traits_ppu.hpp>"]),
    ("cute/atom/mma_traits_sm61.hpp",    ["<cute/arch/mma_sm61.hpp>",  "<cute/atom/mma_traits_ppu.hpp>"]),
    # --- cute/container layer ---
    ("cute/container/cuda_types.hpp",    ["<cute/container/ppu_types.hpp>"]),
    # --- cutlass/arch layer (Step 9) ---
    ("cutlass/arch/memory_sm75.h",      ['"cutlass/arch/memory_ppu.h"']),
    ("cutlass/arch/memory_sm80.h",      ['"cutlass/arch/memory_ppu.h"']),
    ("cutlass/arch/mma_sm60.h",         ['"cutlass/arch/mma_ppu.h"']),
    ("cutlass/arch/mma_sm61.h",         ['"cutlass/arch/mma_ppu.h"']),
    ("cutlass/arch/simd_sm60.h",        ['"cutlass/arch/simd_ppu.h"']),
    ("cutlass/arch/simd_sm61.h",        ['"cutlass/arch/simd_ppu.h"']),
    # --- cutlass/epilogue layer (Step 12) ---
    ("cutlass/epilogue/collective/sm70_epilogue_vectorized.hpp",         ['"cutlass/epilogue/collective/ppu_epilogue_vectorized.hpp"']),
    ("cutlass/epilogue/collective/sm70_epilogue_vectorized_array.hpp",   ['"cutlass/epilogue/collective/ppu_epilogue_vectorized_array.hpp"']),
    ("cutlass/epilogue/collective/sm70_epilogue_vectorized_evt.hpp",     ['"cutlass/epilogue/collective/ppu_epilogue_vectorized_evt.hpp"']),
    ("cutlass/epilogue/collective/sm70_epilogue_vectorized_parallel.hpp",['"cutlass/epilogue/collective/ppu_epilogue_vectorized_parallel.hpp"']),
    ("cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp",          ['"cutlass/epilogue/fusion/ppu_visitor_tma_warpspecialized.hpp"']),
    ("cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp",     ['"cutlass/epilogue/fusion/ppu_visitor_load_tma_warpspecialized.hpp"']),
    ("cutlass/epilogue/fusion/sm90_visitor_store_tma_warpspecialized.hpp",    ['"cutlass/epilogue/fusion/ppu_visitor_store_tma_warpspecialized.hpp"']),
    ("cutlass/epilogue/fusion/sm90_visitor_compute_tma_warpspecialized.hpp",  ['"cutlass/epilogue/fusion/ppu_visitor_compute_tma_warpspecialized.hpp"']),
    # --- cutlass/gemm layer (Step 14) ---
    ("cutlass/gemm/collective/sm80_mma_multistage.hpp",          ['"cutlass/gemm/collective/ppu_mma_multistage.hpp"']),
    ("cutlass/gemm/collective/sm70_mma_twostage_ldmatrix.hpp",   ['"cutlass/gemm/collective/ppu_mma_twostage_ldmatrix.hpp"']),
    ("cutlass/gemm/kernel/sm70_gemm.hpp",                        ['"cutlass/gemm/kernel/ppu_gemm.hpp"']),
    ("cutlass/gemm/kernel/sm90_tile_scheduler.hpp",              ['"cutlass/gemm/kernel/ppu_tile_scheduler.hpp"']),
    ("cutlass/gemm/kernel/sm90_tile_scheduler_group.hpp",        ['"cutlass/gemm/kernel/ppu_tile_scheduler_group.hpp"']),
    ("cutlass/gemm/kernel/sm90_tile_scheduler_stream_k.hpp",     ['"cutlass/gemm/kernel/ppu_tile_scheduler_stream_k.hpp"']),
    ("cutlass/gemm/threadblock/default_mma_core_sm80.h",         ['"cutlass/gemm/threadblock/default_mma_core_ppu.h"']),
    # gemm/thread mma hub wrappers (sm50+sm60+sm61 → ppu)
    ("cutlass/gemm/thread/mma_sm50.h",                           ['"cutlass/gemm/thread/mma_ppu.h"']),
    ("cutlass/gemm/thread/mma_sm60.h",                           ['"cutlass/gemm/thread/mma_ppu.h"']),
    ("cutlass/gemm/thread/mma_sm61.h",                           ['"cutlass/gemm/thread/mma_ppu.h"']),
]


def inject_ppu_aliases(include_dir, changed):
    """Inject CUDA compatibility aliases at the end of PPU source files, wrapped in a separate namespace block."""
    marker = '// === CUDA compatibility aliases ==='
    for rel_path, (ns, alias_body) in PPU_ALIAS_INJECTIONS.items():
        fpath = os.path.join(include_dir, rel_path)
        if not os.path.exists(fpath):
            continue
        with open(fpath, 'r') as f:
            content = f.read()
        if marker in content:
            continue  # already injected, skip
        block = f"\n{marker}\nnamespace {ns} {{\n{alias_body}\n}} // namespace {ns}\n"
        content += block
        with open(fpath, 'w') as f:
            f.write(content)
        changed.append(f"{rel_path} (inject aliases)")


def create_wrappers(include_dir, changed):
    """Generate pure header redirect thin wrappers (no alias content)."""
    for rel_path, includes in WRAPPER_SPECS:
        fpath = os.path.join(include_dir, rel_path)
        if os.path.exists(fpath):
            continue
        with open(fpath, 'w') as f:
            f.write('#pragma once\n')
            f.write(f'// Compatibility wrapper: {rel_path}\n')
            for inc in includes:
                f.write(f'#include {inc}\n')
        changed.append(f"{rel_path} (wrapper)")


def create_arch_map_header(include_dir, changed):
    """Create the HGGC<->CUDA architecture macro mapping header."""
    header_rel = 'cutlass/arch/hggc_arch_map.h'
    header_path = os.path.join(include_dir, header_rel)
    if os.path.exists(header_path):
        return
    content = """\
#pragma once
// HGGC <-> CUDA architecture macro mapping
// The PPU compiler auto-defines __HGGC_ARCH__ (100/150); CUDA/nvcc does not.
// This header defines __HGGC_ARCH__ based on __CUDA_ARCH__.
// Strict mapping: HGGC 100 <-> CUDA 800 (Sm80), HGGC 150 <-> CUDA 890 (Sm89)
// If __HGGC_ARCH__ is already externally defined, validate it matches __CUDA_ARCH__; error on mismatch.

#if defined(__CUDA_ARCH__)
#  if __CUDA_ARCH__ == 800
#    if defined(__HGGC_ARCH__)
#      if __HGGC_ARCH__ != 100
#        error "__HGGC_ARCH__ mismatch: expected 100 for __CUDA_ARCH__ 800"
#      endif
#    else
#      define __HGGC_ARCH__ 100
#    endif
#  elif __CUDA_ARCH__ == 890
#    if defined(__HGGC_ARCH__)
#      if __HGGC_ARCH__ != 150
#        error "__HGGC_ARCH__ mismatch: expected 150 for __CUDA_ARCH__ 890"
#      endif
#    else
#      define __HGGC_ARCH__ 150
#    endif
#  endif
#endif
"""
    os.makedirs(os.path.dirname(header_path), exist_ok=True)
    with open(header_path, 'w') as f:
        f.write(content)
    changed.append(f"{header_rel} (create)")


def inject_arch_map_includes(include_dir, changed):
    """Scan all files that use __HGGC_ARCH__ and inject an include of the architecture mapping header."""
    map_include = '#include "cutlass/arch/hggc_arch_map.h"'
    code_exts = ('.h', '.hpp', '.cuh', '.cu', '.c', '.cpp', '.inl', '.inc')

    for root, dirs, files in os.walk(include_dir):
        for fname in files:
            if not fname.endswith(code_exts):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            except Exception:
                continue
            if '__HGGC_ARCH__' not in content:
                continue
            if map_include in content:
                continue  # already injected, skip

            lines = content.split('\n')
            insert_idx = 0
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped == '#pragma once':
                    insert_idx = i + 1
                    break
                if (stripped.startswith('#define ') and i > 0
                        and lines[i - 1].strip().startswith('#ifndef')):
                    insert_idx = i + 1
                    break
            lines.insert(insert_idx, map_include)
            with open(fpath, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            rel = os.path.relpath(fpath, include_dir)
            changed.append(f"{rel} (arch map include)")

def _actlize_include_dir():
    """Locate the actlize cutlass include dir (mirrors build_lib.py resolution:
    $ACTLIZE_DIR/include first, then the repo-local csrc/actlize/include)."""
    candidates = []
    env_actlize = os.environ.get('ACTLIZE_DIR', '')
    if env_actlize:
        candidates.append(os.path.join(env_actlize, 'include'))
    candidates.append(os.path.join('csrc', 'actlize', 'include'))
    for cand in candidates:
        if os.path.isdir(cand):
            return cand
    return None


# =============================================================================
# Bulk file transformation
# =============================================================================

SUFFIXES = ('.h', '.hpp', '.cu', '.cuh', '.cpp', '.inl')


def find_source_files(directories):
    """Find all source files in given directories."""
    files = []
    for d in directories:
        if not os.path.isdir(d):
            continue
        for root, dirs, filenames in os.walk(d):
            # Prune the nested actlize submodule (cudafied separately via the
            # 'actlize' converter) and any .git dir so a root-level csrc/ walk
            # in the flattened layout does not double-convert them.
            dirs[:] = [x for x in dirs if x not in ('actlize', '.git')]
            for f in filenames:
                if f.endswith(SUFFIXES):
                    files.append(os.path.join(root, f))
    return files


def transform_file(filepath, maps, pattern):
    """Apply reverse transformations to a single file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except (UnicodeDecodeError, UnicodeError):
        return False  # Skip binary or non-UTF-8 files

    # Skip if already in compat mode
    if 'cudaStream_t' in content and '__CUDA_ARCH__' in content and 'hggcStream_t' not in content:
        return False

    new_content = apply_stream_cast_removal(content)
    new_content = replace_content(new_content, maps, pattern)
    new_content = apply_reverse_arch_values(new_content)

    if new_content == content:
        return False

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)
    return True


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Flash-Attention PPU-original -> CUDA-Compatible Transform (no git dependency)')
    parser.add_argument('target_dir', nargs='?', default='.', help='Target flash-attention root directory')
    parser.add_argument('--dry-run', action='store_true', help='Preview without modifying files')
    parser.add_argument('--verbose', action='store_true', help='Print each modified file')
    args = parser.parse_args()

    target_dir = os.path.abspath(args.target_dir)
    # Two accepted layouts: the nested-module base flash-attention checkout
    # (setup.py + csrc/flash_attn), and the flattened flex_flash_attention repo
    # where the module has been promoted to the repo root (build_lib.py +
    # instantiations/, no base flash_attn tree).
    is_base_layout = (os.path.isfile(os.path.join(target_dir, 'setup.py'))
                      and os.path.isdir(os.path.join(target_dir, 'csrc/flash_attn')))
    is_flat_layout = (os.path.isfile(os.path.join(target_dir, 'build_lib.py'))
                      and os.path.isdir(os.path.join(target_dir, 'instantiations')))
    if not (is_base_layout or is_flat_layout):
        print(f"ERROR: {target_dir} is not a flash-attention root directory.")
        sys.exit(1)
    os.chdir(target_dir)

    print("=" * 60)
    print("Flash-Attention PPU-original -> CUDA-Compatible Transform")
    print(f"Target: {target_dir}")
    print("=" * 60)

    t0 = time.time()

    # Step 1: Structural rewrites
    print("\n[1/5] Structural file restorations...")
    n_struct = 0
    if not args.dry_run:
        if rewrite_flash_h():
            n_struct += 1; print("  -> csrc/flash_attn/src/flash.h")
        if rewrite_hardware_info_h():
            n_struct += 1; print("  -> csrc/flash_attn/src/hardware_info.h")
        if rewrite_philox_unpack():
            n_struct += 1; print("  -> csrc/flash_attn/src/philox_unpack.cuh")
        if rewrite_utils_h():
            n_struct += 1; print("  -> csrc/flash_attn/src/utils.h")
        n_struct += rewrite_launch_template_guards()
        print(f"  -> launch template guards restored")
    else:
        print("  [dry-run] Would restore flash.h, hardware_info.h, philox_unpack.cuh, utils.h, launch templates")
    print(f"  Total: {n_struct} files")

    # Step 1b: Delete hggcrt_driver_types.h shim
    print("\n[1b/5] Delete hggcrt_driver_types.h shim...")
    n_shim = 0
    if not args.dry_run:
        n_shim = delete_hggcrt_driver_types_shim()
        if n_shim:
            print(f"  -> Deleted {n_shim} shim file(s)")
    else:
        print("  [dry-run] Would delete hggcrt_driver_types.h shim")

    # Step 2: Bulk reverse replacement (includes hgtx→nvtx, HGGC macros→CUDA macros)
    print("\n[2/5] Bulk HGGC/HGTX -> CUDA/NVTX replacement...")
    maps = get_all_reverse_maps()
    pattern = build_pattern(maps)

    # flex-flash-attention variant: also walk the top-level module dir.
    # find_source_files() skips absent dirs (os.path.isdir guard), so this is a
    # no-op on trees without flex_flash_attention/ (a plain flash-attention
    # checkout), which keeps the base converter's behavior unchanged elsewhere.
    # Flattened layout (module promoted to the repo root, no nested
    # flex_flash_attention/ dir) walks the root csrc/, instantiations/,
    # include/ and hopper/ instead; the nested actlize submodule is pruned in
    # find_source_files().
    if os.path.isdir('flex_flash_attention'):
        source_dirs = ['csrc/flash_attn', 'hopper', 'flex_flash_attention']
    else:
        source_dirs = ['csrc', 'instantiations', 'include', 'hopper']
    files = find_source_files(source_dirs)
    print(f"  Found {len(files)} source files")

    n_transformed = 0
    for f in files:
        if args.dry_run:
            content = open(f).read()
            new = apply_stream_cast_removal(content)
            new = replace_content(new, maps, pattern)
            new = apply_reverse_arch_values(new)
            if new != content:
                n_transformed += 1
                if args.verbose:
                    print(f"    [dry-run] {f}")
        else:
            if transform_file(f, maps, pattern):
                n_transformed += 1
                if args.verbose:
                    print(f"    {f}")
    print(f"  Transformed {n_transformed} files")

    # Step 3: setup.py transformation
    print("\n[3/5] Setup.py transformation...")
    n_setup = 0
    if not args.dry_run:
        n_setup = rewrite_setup_py()
    else:
        print("  [dry-run] Would modify setup.py and hopper/setup.py")
    print(f"  Modified {n_setup} files")

    # Step 3b: self-contained ACTLIZE conversion (naming + CUDA-compat shim).
    # The bulk walk above prunes the actlize submodule, so the flex flow converts
    # its bundled actlize checkout here on its own: apply_actlize_naming_fixups
    # does the base PPU/HGGC -> CUDA naming and the Cu -> Sm renames, then the
    # SM75_/SM80_ aliases + thin wrappers + arch macro map that the kernels depend
    # on are generated into the actlize cutlass include tree.  No dependency on
    # the shared actlize converter.  Idempotent (word-boundary renames +
    # marker/exists guards), so re-runs and a prior actlize conversion are safe.
    print("\n[3b/5] ACTLIZE conversion (naming + CUDA-compat shim)...")
    actlize_inc = _actlize_include_dir()
    n_shim_gen = 0
    if actlize_inc is None:
        print("  WARNING: actlize include dir not found "
              "(csrc/actlize/include or $ACTLIZE_DIR); shim skipped")
    elif args.dry_run:
        print(f"  [dry-run] Would generate SM75/SM80 shim into {actlize_inc}")
    else:
        shim_changed = []
        apply_actlize_naming_fixups(actlize_inc, shim_changed)
        inject_ppu_aliases(actlize_inc, shim_changed)
        create_wrappers(actlize_inc, shim_changed)
        create_arch_map_header(actlize_inc, shim_changed)
        inject_arch_map_includes(actlize_inc, shim_changed)
        n_shim_gen = len(shim_changed)
        if args.verbose:
            for c in sorted(shim_changed):
                print(f"    {c}")
        print(f"  Generated/updated {n_shim_gen} shim entries in {actlize_inc}")

    # Step 4: Summary
    t1 = time.time()
    print(f"\n[4/5] Done in {t1-t0:.1f}s")
    print("=" * 60)
    print(f"  Structural:    {n_struct}")
    print(f"  Shim deleted:  {n_shim}")
    print(f"  Bulk replace:  {n_transformed}")
    print(f"  Setup.py:      {n_setup}")
    print(f"  ACTLIZE shim:  {n_shim_gen}")
    print("=" * 60)

    if args.dry_run:
        print("\n  [DRY RUN - no files modified]")
    else:
        print("\n  Next steps:")
        print("  1. Build FA2: MAX_JOBS=8 python3 setup.py bdist_wheel")
        print("  2. Build FA3: cd hopper && MAX_JOBS=8 python3 setup.py bdist_wheel")


if __name__ == '__main__':
    main()
