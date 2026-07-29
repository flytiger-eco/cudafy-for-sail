#!/usr/bin/env python3
"""
Flash-Attention PPU-original → CUDA-Compatible Transformation Script (v2.7.4).

Transforms flash-attention (FA2) from ppu-original compilation back to
nvcc wrapper / CUDAExtension compilation. Pure text-based conversion, no git dependency.

Usage:
    python3 cuda_compat_v2.7.4.py [TARGET_DIR] [--dry-run] [--verbose]

Run from the flash-attention root directory, or pass it as TARGET_DIR.
"""

import argparse
import os
import re
import sys
import time

# =============================================================================
# Reverse replacement maps (hggc/PPU → cuda/SM)
# =============================================================================

REVERSE_RUNTIME_API = {
    'hggcStream_t': 'cudaStream_t',
    'hggcError_t': 'cudaError_t',
    'hggcSuccess': 'cudaSuccess',
    'hggcGetDevice': 'cudaGetDevice',
    'hggcDeviceGetAttribute': 'cudaDeviceGetAttribute',
    'hggcDevAttrMaxSharedMemoryPerMultiprocessor': 'cudaDevAttrMaxSharedMemoryPerMultiprocessor',
    'hggcDevAttrMaxSharedMemoryPerBlockOptin': 'cudaDevAttrMaxSharedMemoryPerBlockOptin',
    'hggcDevAttrMultiProcessorCount': 'cudaDevAttrMultiProcessorCount',
    'hggcDevAttrComputeCapabilityMajor': 'cudaDevAttrComputeCapabilityMajor',
    'hggcDevAttrComputeCapabilityMinor': 'cudaDevAttrComputeCapabilityMinor',
    'hggcFuncSetAttribute': 'cudaFuncSetAttribute',
    'hggcFuncAttributeMaxDynamicSharedMemorySize': 'cudaFuncAttributeMaxDynamicSharedMemorySize',
    'hggcOccupancyMaxActiveBlocksPerMultiprocessor': 'cudaOccupancyMaxActiveBlocksPerMultiprocessor',
    'hggcGetErrorString': 'cudaGetErrorString',
    'hggcGetLastError': 'cudaGetLastError',
}

REVERSE_HEADER = {
    '<hggc_fp16.h>': '<cuda_fp16.h>',
    '<hggc_bf16.h>': '<cuda_bf16.h>',
    '<hggc_runtime.h>': '<cuda_runtime.h>',
    '"hggc_runtime.h"': '"cuda_runtime.h"',
    '<hgtx3/hgToolsExt.h>': '<nvtx3/nvToolsExt.h>',
}

REVERSE_MACRO = {
    # NOTE: __HGGC_ARCH__ is NOT globally replaced here.
    # It's handled specially in apply_reverse_arch_macro() to preserve
    # PPU-specific uses like __HGGC_ARCH__ == 100 and == 10500.
    '__HGGC_NO_HALF_OPERATORS__': '__CUDA_NO_HALF_OPERATORS__',
    '__HGGC_NO_HALF_CONVERSIONS__': '__CUDA_NO_HALF_CONVERSIONS__',
    '__HGGC_NO_HALF2_OPERATORS__': '__CUDA_NO_HALF2_OPERATORS__',
    '__HGGC_NO_BFLOAT16_CONVERSIONS__': '__CUDA_NO_BFLOAT16_CONVERSIONS__',
    'hgtxDomainHandle_t': 'nvtxDomainHandle_t',
    'hgtxEventAttributes_t': 'nvtxEventAttributes_t',
    'HGTX_VERSION': 'NVTX_VERSION',
    'HGTX_MESSAGE_TYPE_ASCII': 'NVTX_MESSAGE_TYPE_ASCII',
    'hgtxDomainCreateA': 'nvtxDomainCreateA',
    'hgtxDomainDestroy': 'nvtxDomainDestroy',
    'hgtxDomainRangePushEx': 'nvtxDomainRangePushEx',
    'hgtxDomainRangePop': 'nvtxDomainRangePop',
    'use_hgtx_': 'use_nvtx_',
    'use_hgtx': 'use_nvtx',
}

# Applied only on lines containing __CUDA_ARCH__ (after macro replacement)
REVERSE_ARCH_VALUE = {
    '>= 100': '>= 800',
    '== 100': '== 800',
    '== 150': '== 890',
}


def get_all_reverse_maps():
    """Merge all reverse maps, sorted longest-first."""
    merged = {}
    merged.update(REVERSE_RUNTIME_API)
    merged.update(REVERSE_HEADER)
    merged.update(REVERSE_MACRO)
    return merged


# =============================================================================
# Replacement engine
# =============================================================================

def build_pattern(maps):
    """Build compiled regex from maps dict, longest key first."""
    sorted_keys = sorted(maps.keys(), key=lambda x: -len(x))
    pattern = re.compile("|".join(re.escape(k) for k in sorted_keys))
    return pattern


def replace_content(content, maps, pattern):
    """Apply dict-based replacement."""
    return pattern.sub(lambda m: maps[m.group(0)], content)


def apply_reverse_arch_values(content, filepath=''):
    """Replace __HGGC_ARCH__ -> __CUDA_ARCH__ with arch value mapping.

    Only == 10500 is unconditionally preserved (PPU-internal).
    For == 100: converted in hopper/ files (originally __CUDA_ARCH__ == 800),
    but preserved in csrc/flash_attn/src/ files (originally __HGGC_ARCH__ == 100).
    """
    # Determine if this file's == 100 should be converted
    # hopper/ files originally used __CUDA_ARCH__ == 800, csrc/ files used __HGGC_ARCH__ == 100
    is_hopper_file = 'hopper/' in filepath.replace(os.sep, '/')
    convert_eq_100 = is_hopper_file
    lines = content.split('\n')
    new_lines = []
    for line in lines:
        if '__HGGC_ARCH__' in line:
            # Check if this is PPU-internal-only (== 10500) that should never be converted
            if '== 10500' in line:
                # PPU-internal arch value, keep as __HGGC_ARCH__
                pass
            elif '>= 100' in line or '>= 150' in line:
                # Generic arch check: convert to __CUDA_ARCH__ >= 800/890
                line = line.replace('__HGGC_ARCH__', '__CUDA_ARCH__')
                for old, new in REVERSE_ARCH_VALUE.items():
                    line = line.replace(old, new)
            elif '>= 900' in line or '<= 890' in line:
                # SM90+ or <=SM89 check: just swap macro name, value stays same
                line = line.replace('__HGGC_ARCH__', '__CUDA_ARCH__')
            elif '== 150' in line:
                # PPU 1.5 dispatch: convert to __CUDA_ARCH__ == 890
                line = line.replace('__HGGC_ARCH__', '__CUDA_ARCH__')
                line = line.replace('== 150', '== 890')
            elif '== 100' in line:
                if convert_eq_100:
                    # SM80 dispatch: convert to __CUDA_ARCH__ == 800
                    line = line.replace('__HGGC_ARCH__', '__CUDA_ARCH__')
                    line = line.replace('== 100', '== 800')
                # else: csrc/ file, keep as __HGGC_ARCH__ == 100
            else:
                # Other patterns (> 10000, etc): just swap macro name
                line = line.replace('__HGGC_ARCH__', '__CUDA_ARCH__')
        elif '__CUDA_ARCH__' in line:
            # Already converted by other means, apply arch value mapping
            for old, new in REVERSE_ARCH_VALUE.items():
                line = line.replace(old, new)
        new_lines.append(line)
    return '\n'.join(new_lines)


# =============================================================================
# Structural rewrites (pure text, no git)
# =============================================================================

def restore_flash_h():
    """Restore csrc/flash_attn/src/flash.h: remove __HGGCCC__ block, add cuda includes."""
    fp = 'csrc/flash_attn/src/flash.h'
    if not os.path.isfile(fp):
        return False
    content = open(fp).read()
    if '#ifdef __HGGCCC__' not in content:
        return False

    # Replace the entire #ifdef __HGGCCC__ ... #endif block + #include <vector>
    # with original CUDA includes
    pat = re.compile(
        r'#ifdef __HGGCCC__\n.*?#endif\s*\n\s*#include <vector>',
        re.DOTALL)
    replacement = ('#include <cuda.h>\n#include <vector>\n\n'
                   '#include <ATen/cuda/CUDAGeneratorImpl.h> // For at::Generator and at::PhiloxCudaState')
    new_content = pat.sub(replacement, content)

    if new_content == content:
        return False
    open(fp, 'w').write(new_content)
    return True


def restore_hardware_info_h():
    """Rewrite csrc/flash_attn/src/hardware_info.h to standard CUDA version."""
    fp = 'csrc/flash_attn/src/hardware_info.h'
    if not os.path.isfile(fp):
        return False
    content = open(fp).read()
    if '#ifdef __HGGCCC__' not in content and 'hggcGetDevice' not in content:
        return False

    new_content = """/******************************************************************************
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


def restore_philox_unpack():
    """Rewrite csrc/flash_attn/src/philox_unpack.cuh to standard version."""
    fp = 'csrc/flash_attn/src/philox_unpack.cuh'
    if not os.path.isfile(fp):
        return False
    content = open(fp).read()
    if '#ifdef __HGGCCC__' not in content:
        return False

    new_content = """// This is purely so that it works with torch 2.1. For torch 2.2+ we can include ATen/cuda/PhiloxUtils.cuh
#pragma once
#include <ATen/cuda/detail/UnpackRaw.cuh>
"""
    open(fp, 'w').write(new_content)
    return True


def restore_utils_h():
    """Restore csrc/flash_attn/src/utils.h fp16/bf16 includes."""
    fp = 'csrc/flash_attn/src/utils.h'
    if not os.path.isfile(fp):
        return False
    content = open(fp).read()
    if 'USE_CLANG' not in content and 'hggc_fp16' not in content:
        return False

    # Replace the USE_CLANG/hggc guarded block with original CUDA includes
    old_pat = re.compile(
        r'#if defined\(USE_CLANG\)\n#include <hggc_fp16\.h>\n#include <hggc_bf16\.h>\n#else\n.*?#endif\n#endif',
        re.DOTALL)
    new_block = '#include <cuda_fp16.h>\n\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800\n#include <cuda_bf16.h>\n#endif'
    new_content = old_pat.sub(new_block, content)

    if new_content == content:
        # Try alternate pattern (single #endif version)
        old_pat2 = re.compile(
            r'#if defined\(USE_CLANG\)\n#include <hggc_fp16\.h>\n#include <hggc_bf16\.h>\n#else\n#include <cuda_fp16\.h>.*?#endif',
            re.DOTALL)
        new_content = old_pat2.sub(new_block, content)

    if new_content == content:
        return False
    open(fp, 'w').write(new_content)
    return True


def restore_launch_template_guards():
    """Remove #ifndef __HGGCCC__ guards from launch templates, restore c10/ATen includes."""
    targets = [
        ('csrc/flash_attn/src/flash_fwd_launch_template.h',
         '#ifndef __HGGCCC__\n#include <c10/cuda/CUDAException.h>\n#include <ATen/cuda/CUDAContext.h>\n#else\n#define C10_CUDA_CHECK(x) (void)(x)\n#define C10_CUDA_KERNEL_LAUNCH_CHECK()\n#endif',
         '#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK\n#include <ATen/cuda/CUDAContext.h>'),
        ('csrc/flash_attn/src/flash_bwd_launch_template.h',
         '#ifndef __HGGCCC__\n#include <c10/cuda/CUDAException.h>\n#else\n#define C10_CUDA_CHECK(x) (void)(x)\n#define C10_CUDA_KERNEL_LAUNCH_CHECK()\n#endif',
         '#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK'),
    ]
    count = 0
    for fp, old, new in targets:
        if not os.path.isfile(fp):
            continue
        content = open(fp).read()
        if old in content:
            open(fp, 'w').write(content.replace(old, new))
            count += 1
    return count


def delete_hggcrt_driver_types_shim():
    """Delete the hggcrt_driver_types.h shim files created for ppu-original mode."""
    shims = [
        'csrc/flash_attn/src/hggcrt_driver_types.h',
        'hopper/hggcrt_driver_types.h',
    ]
    count = 0
    for fp in shims:
        if os.path.isfile(fp):
            os.remove(fp)
            count += 1
    return count


def restore_flash_api_cpp():
    """Restore flash_api.cpp stream casts via pure text replacement (no git).

    In ppu-original mode: hggcStream_t stream = (hggcStream_t)at::cuda::getCurrentCUDAStream().stream();
    In ppu-original mode: auto stream = at::cuda::getCurrentCUDAStream().stream();

    Function signatures: hggcStream_t → cudaStream_t (handled by bulk replacement)
    """
    targets = ['csrc/flash_attn/flash_api.cpp', 'hopper/flash_api.cpp']
    count = 0
    for fp in targets:
        if not os.path.isfile(fp):
            continue
        content = open(fp).read()
        if 'hggcStream_t' not in content:
            continue

        # Replace the explicit cast pattern
        new_content = re.sub(
            r'hggcStream_t\s+stream\s*=\s*\(hggcStream_t\)at::cuda::getCurrentCUDAStream\(\)\.stream\(\);',
            'auto stream = at::cuda::getCurrentCUDAStream().stream();',
            content)

        if new_content != content:
            open(fp, 'w').write(new_content)
            count += 1
    return count


# =============================================================================
# setup.py transformation (pure text, no git)
# =============================================================================

def transform_setup_py():
    """Transform setup.py from HGCCBuildExtension back to CUDAExtension.

    For FA2 setup.py: full regeneration with extracted sources/includes.
    For hopper/setup.py: in-place patching (replace HGCCBuildExtension with CUDAExtension logic).
    """
    count = 0

    # FA2 setup.py: full regeneration
    fp = 'setup.py'
    if os.path.isfile(fp) and 'HGCCBuildExtension' in open(fp).read():
        content = open(fp).read()
        sources_match = re.search(r'sources=\[(.*?)\]', content, re.DOTALL)
        include_match = re.search(r'include_dirs=\[(.*?)\]', content, re.DOTALL)
        name_match = re.search(r'name=["\']([^"\']+)["\']', content)
        if sources_match:
            sources_block = sources_match.group(1)
            include_block = include_match.group(1) if include_match else ''
            ext_name = name_match.group(1) if name_match else 'flash_attn_2_cuda'
            new_content = generate_cuda_extension_setup(ext_name, sources_block, include_block, fp)
            open(fp, 'w').write(new_content)
            count += 1
        else:
            print(f"  WARNING: Cannot extract sources from {fp}, skipping")

    # hopper/setup.py: in-place patching
    fp = 'hopper/setup.py'
    if os.path.isfile(fp) and 'HGCCBuildExtension' in open(fp).read():
        if _patch_hopper_setup(fp):
            count += 1
    return count


def _patch_hopper_setup(fp):
    """Patch hopper/setup.py in-place: HGCCBuildExtension → CUDAExtension."""
    content = open(fp).read()

    # 1. Remove the entire HGCCBuildExtension class definition
    content = re.sub(
        r'# =+\n# PPU HGCC Build Extension\n# =+\n\nclass HGCCBuildExtension.*?(?=\n# =)',
        '', content, flags=re.DOTALL)

    # 2. Add CUDAExtension import
    content = content.replace(
        'from setuptools import setup, find_packages, Extension\nfrom setuptools.command.build_ext import build_ext',
        'from setuptools import setup, find_packages\nfrom torch.utils.cpp_extension import BuildExtension, CUDAExtension')

    # 3. Replace Extension( with CUDAExtension(
    content = content.replace(
        '    ext_modules.append(\n        Extension(',
        '    ext_modules.append(\n        CUDAExtension(')

    # 4. Replace extra_compile_args with nvcc format
    # NOTE: this must mirror the ppu-original build (hopper/setup.py
    # nvcc_flags/cc_flag/feature_args composition. cc_flag is empty here because
    # DISABLE_SM90 is hardcoded True in this PPU-only build (SM90 removed), which
    # matches the original's behavior when DISABLE_SM90 is True (no -gencode emitted).
    content = re.sub(
        r'extra_compile_args=\{\s*"hgcc_extra": feature_args,\s*"cxx_extra": feature_args,\s*\}',
        '''extra_compile_args={
            "cxx": ["-O3", "-std=c++17"] + feature_args,
            "nvcc": [
                "--threads", "4",
                "-O3", "-std=c++17",
                "--ftemplate-backtrace-limit=0",
                "--use_fast_math",
                "--resource-usage",
                "-lineinfo",
                "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",
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
                "-mllvm", "-ppu-force-vregrr=true",
                "-Xfatbin", "--compress-all",
                "-Xfatbin", "--compress-all",
            ] + feature_args,
        }''',
        content)

    # 5. Replace cmdclass HGCCBuildExtension → BuildExtension
    content = content.replace(
        '"build_ext": HGCCBuildExtension',
        '"build_ext": BuildExtension')

    # NOTE: os.environ["HGGC_ENABLE_COMPRESS"] = "1" is intentionally KEPT (not
    # removed) because the original 43af4cc:hopper/setup.py also sets this exact
    # env var (paired with the -Xfatbin --compress-all flags above).

    # 6. Add PPU_OPTION env var (missing in ppu-original source, present in original 43af4cc)
    content = content.replace(
        '    os.environ["HGGC_ENABLE_COMPRESS"] = "1"\n',
        '    os.environ["HGGC_ENABLE_COMPRESS"] = "1"\n    os.environ["PPU_OPTION"] = "-no-hggc-embed-bc"\n')

    open(fp, 'w').write(content)
    return True


def generate_cuda_extension_setup(ext_name, sources_block, include_block, filepath):
    """Generate a standard CUDAExtension setup.py for FA2/FA3."""
    is_fa3 = 'hopper' in filepath

    if is_fa3:
        return _generate_fa3_setup(ext_name, sources_block, include_block)
    else:
        return _generate_fa2_setup(ext_name, sources_block, include_block)


def _generate_fa2_setup(ext_name, sources_block, include_block):
    return f'''# Copyright (c) 2023, Tri Dao.
import sys
import functools
import warnings
import os
import re
import ast
import glob
import shutil
from pathlib import Path
from packaging.version import parse, Version

from setuptools import setup, find_packages
import subprocess

from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

import torch
from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
    CUDA_HOME,
)

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

this_dir = os.path.dirname(os.path.abspath(__file__))
USE_PPU = 'PPU_SDK' in os.environ.keys()

PACKAGE_NAME = "flash_attn"

SKIP_CUDA_BUILD = os.getenv("FLASH_ATTENTION_SKIP_CUDA_BUILD", "FALSE") == "TRUE"


@functools.lru_cache(maxsize=None)
def cuda_archs() -> str:
    return os.getenv("FLASH_ATTN_CUDA_ARCHS", "80;90;100;120").split(";")


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])
    return raw_output, bare_metal_version


def get_package_version():
    with open(Path(this_dir) / "flash_attn" / "__init__.py", "r") as f:
        version_match = re.search(r"^__version__\\s*=\\s*(.*)$", f.read(), re.MULTILINE)
    public_version = ast.literal_eval(version_match.group(1))
    local_version = os.environ.get("FLASH_ATTN_LOCAL_VERSION")
    if local_version:
        return f"{{public_version}}+{{local_version}}"
    else:
        return str(public_version)


class CachedWheelsCommand(_bdist_wheel):
    def run(self):
        return super().run()


class NinjaBuildExtension(BuildExtension):
    def __init__(self, *args, **kwargs) -> None:
        if not os.environ.get("MAX_JOBS"):
            import psutil
            max_num_jobs_cores = max(1, os.cpu_count() // 2)
            free_memory_gb = psutil.virtual_memory().available / (1024 ** 3)
            max_num_jobs_memory = int(free_memory_gb / 9)
            max_jobs = max(1, min(max_num_jobs_cores, max_num_jobs_memory))
            os.environ["MAX_JOBS"] = str(max_jobs)
        super().__init__(*args, **kwargs)


cmdclass = {{}}
ext_modules = []

dir_actlize = this_dir + "/csrc/actlize"
if not os.path.exists(dir_actlize):
    repo_actlize = os.path.dirname(this_dir) + "/actlize"
    if not os.path.exists(repo_actlize):
        raise RuntimeError(
            f"actlize does not exist: actlize must be fetched in advance as:\\n"
            f' "{{repo_actlize}}" or "{{dir_actlize}}"'
        )
    else:
        os.symlink(repo_actlize, dir_actlize)

if not SKIP_CUDA_BUILD:
    print("\\n\\ntorch.__version__  = {{}}\\n\\n".format(torch.__version__))

    cc_flag = []
    if "80" in cuda_archs():
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_80,code=sm_80")
    if CUDA_HOME is not None:
        _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
        if bare_metal_version >= Version("11.8") and "90" in cuda_archs():
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_90,code=sm_90")
        if bare_metal_version >= Version("12.8") and "100" in cuda_archs():
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_100,code=sm_100")
        if bare_metal_version >= Version("12.8") and "120" in cuda_archs():
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_120,code=sm_120")

    ext_modules.append(
        CUDAExtension(
            name="{ext_name}",
            sources=[
{sources_block}
            ],
            extra_compile_args={{
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "--threads", "2",
                    "-O3", "-std=c++17",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-mllvm", "-ppu-max-vreg-count=256",
                    "-mllvm", "-ppu-sink-matrix-addr=true",
                    "-mllvm", "-ppu-max-alloca-byte-size=320",
                    "-mllvm", "-ppu-sink-async-addr=true",
                    "-mllvm", "-ppu-sink-load-addr=true",
                    "-mllvm", "-ppu-sink-store-addr=true",
                    "-mllvm", "-ppu-alloca-half-ldst-simplify=true",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                    "-DUSE_PPU",
                    "-DUSE_AIU=1",
                ]
                + cc_flag,
            }},
            include_dirs=[
{include_block}
            ],
        )
    )

setup(
    name=PACKAGE_NAME,
    version=get_package_version(),
    packages=find_packages(
        exclude=(
            "build", "csrc", "include", "tests", "dist", "docs", "benchmarks",
            "flash_attn.egg-info",
        )
    ),
    author="Tri Dao",
    author_email="tri@tridao.me",
    description="Flash Attention: Fast and Memory-Efficient Exact Attention",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Dao-AILab/flash-attention",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
    ],
    ext_modules=ext_modules,
    cmdclass={{"bdist_wheel": CachedWheelsCommand, "build_ext": NinjaBuildExtension}}
    if ext_modules
    else {{"bdist_wheel": CachedWheelsCommand}},
    python_requires=">=3.9",
    install_requires=["torch", "einops"],
    setup_requires=["packaging", "psutil", "ninja"],
)
'''


def _generate_fa3_setup(ext_name, sources_block, include_block):
    return f'''import os
import re
import ast
import glob
import shutil
from pathlib import Path
from packaging.version import parse, Version

from setuptools import setup, find_packages
import subprocess

import torch
from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
    CUDA_HOME,
)

this_dir = os.path.dirname(os.path.abspath(__file__))
USE_PPU = 'PPU_SDK' in os.environ.keys()

PACKAGE_NAME = "flash_attn_3"
VERSION = "3.0.0b1"

cmdclass = {{}}
ext_modules = []

ext_modules.append(
    CUDAExtension(
        name="flash_attn_3._C",
        sources=[
{sources_block}
        ],
        extra_compile_args={{
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": [
                "--threads", "4",
                "-O3", "-std=c++17",
                "--ftemplate-backtrace-limit=0",
                "--use_fast_math",
                "--resource-usage",
                "-lineinfo",
                "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",
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
                "-mllvm", "-ppu-force-vregrr=true",
                "-Xfatbin", "--compress-all",
                "-Xfatbin", "--compress-all",
                "-DUSE_PPU",
                "-DUSE_AIU=1",
            ],
        }},
        include_dirs=[
{include_block}
        ],
    )
)

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={{"build_ext": BuildExtension}},
    python_requires=">=3.9",
    install_requires=["torch", "einops"],
)
'''


# =============================================================================
# Bulk file transformation
# =============================================================================

SUFFIXES = ('.h', '.hpp', '.cu', '.cuh', '.cpp', '.inl', '.py')


def find_source_files(directories):
    """Find all source files in given directories."""
    files = []
    for d in directories:
        if not os.path.isdir(d):
            continue
        for root, _, filenames in os.walk(d):
            # Skip actlize submodule (handled separately by cuda_compat.py)
            if 'actlize' in root:
                continue
            for f in filenames:
                if f.endswith(SUFFIXES):
                    files.append(os.path.join(root, f))
    return files


def transform_file(filepath, maps, pattern):
    """Apply reverse transformations to a single file."""
    with open(filepath, 'r') as f:
        content = f.read()

    # Skip if already in compat mode
    if 'cudaStream_t' in content and '__CUDA_ARCH__' in content and 'hggcStream_t' not in content:
        return False

    new_content = replace_content(content, maps, pattern)
    new_content = apply_reverse_arch_values(new_content, filepath)

    if new_content == content:
        return False

    with open(filepath, 'w') as f:
        f.write(new_content)
    return True


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Flash-Attention PPU-original → CUDA-Compatible Transform')
    parser.add_argument('target_dir', nargs='?', default='.', help='Target flash-attention root directory')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without modifying files')
    parser.add_argument('--verbose', action='store_true', help='Print each modified file')
    args = parser.parse_args()

    target_dir = os.path.abspath(args.target_dir)
    if not os.path.isfile(os.path.join(target_dir, 'setup.py')) or not os.path.isdir(os.path.join(target_dir, 'csrc/flash_attn')):
        print(f"ERROR: {target_dir} is not a flash-attention root directory.")
        sys.exit(1)
    os.chdir(target_dir)

    print("=" * 60)
    print("Flash-Attention PPU-original → CUDA-Compatible Transform")
    print("  (pure text conversion, no git dependency)")
    print(f"Target: {target_dir}")
    print("=" * 60)

    t0 = time.time()

    # Step 1: Structural rewrites
    print("\n[1/4] Structural file restorations...")
    n_struct = 0
    if not args.dry_run:
        if restore_flash_h():
            n_struct += 1; print("  -> csrc/flash_attn/src/flash.h")
        if restore_hardware_info_h():
            n_struct += 1; print("  -> csrc/flash_attn/src/hardware_info.h")
        if restore_philox_unpack():
            n_struct += 1; print("  -> csrc/flash_attn/src/philox_unpack.cuh")
        if restore_utils_h():
            n_struct += 1; print("  -> csrc/flash_attn/src/utils.h")
        n_struct += restore_launch_template_guards()
        print(f"  -> launch templates")
    else:
        print("  [dry-run] Would restore flash.h, hardware_info.h, philox_unpack.cuh, utils.h, launch templates")
    print(f"  Total: {n_struct} files")

    # Step 1b: Delete shim files
    n_shim = 0
    if not args.dry_run:
        n_shim = delete_hggcrt_driver_types_shim()
        if n_shim:
            print(f"  -> Deleted {n_shim} hggcrt_driver_types.h shim(s)")
    else:
        print("  [dry-run] Would delete hggcrt_driver_types.h shims")

    # Step 2: flash_api.cpp stream cast restoration
    print("\n[2/4] flash_api.cpp stream cast restoration...")
    n_api = 0
    if not args.dry_run:
        n_api = restore_flash_api_cpp()
    else:
        print("  [dry-run] Would restore stream casts in flash_api.cpp")
    print(f"  Modified: {n_api} files")

    # Step 3: Bulk reverse replacement
    print("\n[3/4] Bulk HGGC → CUDA replacement...")
    maps = get_all_reverse_maps()
    pattern = build_pattern(maps)

    source_dirs = ['csrc/flash_attn', 'hopper']
    files = find_source_files(source_dirs)
    print(f"  Found {len(files)} source files")

    n_transformed = 0
    for f in files:
        if args.dry_run:
            content = open(f).read()
            new = replace_content(content, maps, pattern)
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
    print(f"  Transformed: {n_transformed} files")

    # Step 4: setup.py transformation
    print("\n[4/4] Setup.py transformation...")
    n_setup = 0
    if not args.dry_run:
        n_setup = transform_setup_py()
    else:
        print("  [dry-run] Would regenerate setup.py as CUDAExtension")
    print(f"  Modified: {n_setup} files")

    # Summary
    t1 = time.time()
    print(f"\n{'=' * 60}")
    print(f"Done in {t1-t0:.1f}s")
    print(f"  Structural:     {n_struct}")
    print(f"  flash_api.cpp:  {n_api}")
    print(f"  Bulk replace:   {n_transformed}")
    print(f"  Setup.py:       {n_setup}")
    print(f"{'=' * 60}")

    if args.dry_run:
        print("\n  [DRY RUN - no files modified]")
    else:
        print("\n  Next: python setup.py bdist_wheel")


if __name__ == '__main__':
    main()
