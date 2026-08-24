#!/usr/bin/env python3
"""
Convert xFormers v0.0.27 from PPU-original naming to CUDA-compatible naming.

The conversion is text-only and does not depend on Git. Convert the bundled
ACTLIZE headers separately with actlize/cuda_compat_v0.5.0.py.

COMPATIBLE_ARCH must map to __CUDA_ARCH__: both expose 800/890 in compatible
builds, while leaving COMPATIBLE_ARCH under nvcc turns guarded autogen kernels
into empty functions. PPU assembly, intrinsics, architecture tags, and compiler
flags remain unchanged because the CUDA-compatible compiler still targets PPU.

Usage:
    python3 cuda_compat_v0.0.27.py [TARGET_DIR] [--dry-run] [--verbose]
"""

import argparse
import os
import re
import sys
import time

CONVERT_RUNTIME_API = {
    'hggcOccupancyMaxActiveBlocksPerMultiprocessor': 'cudaOccupancyMaxActiveBlocksPerMultiprocessor',
    'hggcFuncAttributeMaxDynamicSharedMemorySize': 'cudaFuncAttributeMaxDynamicSharedMemorySize',
    'hggcDevAttrMaxSharedMemoryPerMultiprocessor': 'cudaDevAttrMaxSharedMemoryPerMultiprocessor',
    'hggcDevAttrMaxSharedMemoryPerBlockOptin': 'cudaDevAttrMaxSharedMemoryPerBlockOptin',
    'hggcDevAttrComputeCapabilityMajor': 'cudaDevAttrComputeCapabilityMajor',
    'hggcDevAttrComputeCapabilityMinor': 'cudaDevAttrComputeCapabilityMinor',
    'hggcDevAttrMultiProcessorCount': 'cudaDevAttrMultiProcessorCount',
    'hggcGetDeviceProperties': 'cudaGetDeviceProperties',
    'hggcDeviceGetAttribute': 'cudaDeviceGetAttribute',
    'hggcDeviceSynchronize': 'cudaDeviceSynchronize',
    'hggcEventElapsedTime': 'cudaEventElapsedTime',
    'hggcEventSynchronize': 'cudaEventSynchronize',
    'hggcErrorInvalidValue': 'cudaErrorInvalidValue',
    'hggcFuncGetAttributes': 'cudaFuncGetAttributes',
    'hggcStreamSynchronize': 'cudaStreamSynchronize',
    'hggcFuncSetAttribute': 'cudaFuncSetAttribute',
    'hggcGetErrorString': 'cudaGetErrorString',
    'hggcFuncAttributes': 'cudaFuncAttributes',
    'hggcEventElapsed': 'cudaEventElapsed',
    'hggcEventDestroy': 'cudaEventDestroy',
    'hggcGetLastError': 'cudaGetLastError',
    'hggcMemsetAsync': 'cudaMemsetAsync',
    'hggcEventRecord': 'cudaEventRecord',
    'hggcEventCreate': 'cudaEventCreate',
    'hggcDeviceProp': 'cudaDeviceProp',
    'hggcGetDevice': 'cudaGetDevice',
    'hggcStream_t': 'cudaStream_t',
    'hggcError_t': 'cudaError_t',
    'hggcEvent_t': 'cudaEvent_t',
    'hggcSuccess': 'cudaSuccess',
    'hggcMemset': 'cudaMemset',
    'hggcMalloc': 'cudaMalloc',
    'hggcMemcpy': 'cudaMemcpy',
    'hggcFree': 'cudaFree',
    'hggcError': 'cudaError',
}

CONVERT_HEADER = {
    '<hggc_runtime_api.h>': '<cuda_runtime_api.h>',
    '<hggc_runtime.h>': '<cuda_runtime.h>',
    '"hggc_runtime.h"': '"cuda_runtime.h"',
    '<hggc_fp16.h>': '<cuda_fp16.h>',
    '<hggc_bf16.h>': '<cuda_bf16.h>',
    '<hggc.h>': '<cuda.h>',
    '"hggc.h"': '"cuda.h"',
    'hggc/std/type_traits': 'cuda/std/type_traits',
    'hggc/std/cstdint': 'cuda/std/cstdint',
    'hggc/std/cstddef': 'cuda/std/cstddef',
    'hggc/std/utility': 'cuda/std/utility',
    'hggc/std/cassert': 'cuda/std/cassert',
    'hggc/std/limits': 'cuda/std/limits',
    'hggc/std/tuple': 'cuda/std/tuple',
}

CONVERT_MACRO = {
    '__COMPATIBLECC_VER_MAJOR__': '__CUDACC_VER_MAJOR__',
    '__COMPATIBLECC_VER_MINOR__': '__CUDACC_VER_MINOR__',
    '__COMPATIBLECC_VER_BUILD__': '__CUDACC_VER_BUILD__',
    '__HGGCCC_VER_MAJOR__': '__CUDACC_VER_MAJOR__',
    '__HGGCCC_VER_MINOR__': '__CUDACC_VER_MINOR__',
    '__HGGCCC_RTC__': '__CUDACC_RTC__',
    '__HGGCCC__': '__CUDACC__',
    'COMPATIBLE_ARCH_LIST': '__CUDA_ARCH_LIST__',
    'COMPATIBLE_ARCH': '__CUDA_ARCH__',
    '__HGGC_NO_HALF_OPERATORS__': '__CUDA_NO_HALF_OPERATORS__',
    '__HGGC_NO_HALF_CONVERSIONS__': '__CUDA_NO_HALF_CONVERSIONS__',
    '__HGGC_NO_HALF2_OPERATORS__': '__CUDA_NO_HALF2_OPERATORS__',
    '__HGGC_NO_BFLOAT16_CONVERSIONS__': '__CUDA_NO_BFLOAT16_CONVERSIONS__',
}

CONVERT_LIBRARY = {
    'acrand_kernel.h': 'curand_kernel.h',
    'acrand': 'curand',
}


def get_all_convert_maps():
    merged = {}
    for mapping in (
            CONVERT_RUNTIME_API, CONVERT_HEADER, CONVERT_MACRO,
            CONVERT_LIBRARY):
        merged.update(mapping)
    return merged


def build_pattern(maps):
    keys = sorted(maps, key=len, reverse=True)
    return re.compile("|".join(re.escape(key) for key in keys))


# Kernel generators embed compiler guards in generated CUDA source.
SUFFIXES = ('.h', '.hpp', '.cu', '.cuh', '.cpp', '.inl', '.py')
SOURCE_DIRS = ['xformers/csrc']

# Intentional PPU-only names without CUDA equivalents.
KNOWN_RESIDUAL_TOKENS = (
    'HGGC_MMA_FRAGMENT_VALUE',
    'HGGC_PROFILE_MODE',
)
RESIDUAL_PATTERN = re.compile(
    r'(?i:hggc|acrand)|COMPATIBLE_ARCH|__COMPATIBLECC')


def find_source_files(directories):
    files = []
    for directory in directories:
        if not os.path.isdir(directory):
            continue
        for root, _, filenames in os.walk(directory):
            files.extend(
                os.path.join(root, filename)
                for filename in filenames
                if filename.endswith(SUFFIXES)
            )
    return files


def _read_text(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as source:
            return source.read()
    except UnicodeError:
        return None


def transform_file(filepath, maps, pattern, dry_run=False):
    content = _read_text(filepath)
    if content is None:
        return False

    new_content = pattern.sub(lambda match: maps[match.group(0)], content)
    if new_content == content:
        return False

    if not dry_run:
        with open(filepath, 'w', encoding='utf-8') as destination:
            destination.write(new_content)
    return True


def check_residual_naming(directories):
    residual = []
    for filepath in find_source_files(directories):
        content = _read_text(filepath)
        if content is None:
            continue
        for token in KNOWN_RESIDUAL_TOKENS:
            content = content.replace(token, '')
        if RESIDUAL_PATTERN.search(content):
            residual.append(filepath)
    return residual


def main():
    parser = argparse.ArgumentParser(
        description='xFormers PPU-original -> CUDA-Compatible Transform (no git dependency)')
    parser.add_argument('target_dir', nargs='?', default='.', help='Target xformers root directory')
    parser.add_argument('--dry-run', action='store_true', help='Preview without modifying files')
    parser.add_argument('--verbose', action='store_true', help='Print each modified file')
    args = parser.parse_args()

    target_dir = os.path.abspath(args.target_dir)
    if (not os.path.isfile(os.path.join(target_dir, 'setup.py'))
            or not os.path.isdir(os.path.join(target_dir, 'xformers/csrc/attention/cuda/fmha'))):
        print(f"ERROR: {target_dir} is not an xformers root directory.")
        return 1
    os.chdir(target_dir)

    print("=" * 60)
    print("xFormers PPU-original -> CUDA-Compatible Transform")
    print(f"Target: {target_dir}")
    print("=" * 60)

    started_at = time.time()
    maps = get_all_convert_maps()
    pattern = build_pattern(maps)
    files = find_source_files(SOURCE_DIRS)

    print("\n[1/2] Bulk HGGC/acrand -> CUDA/cuRAND naming replacement...")
    print(f"  Found {len(files)} source files")
    files_modified = 0
    for filepath in files:
        if transform_file(filepath, maps, pattern, args.dry_run):
            files_modified += 1
            if args.verbose:
                prefix = "[dry-run] " if args.dry_run else ""
                print(f"    {prefix}{filepath}")
    print(f"  Transformed {files_modified} files")

    # setup.py is already CUDAExtension-compatible; PPU flags are accepted by nvcc.
    print("\n[2/2] Residual PPU naming check...")
    if args.dry_run:
        print("  [dry-run] skipped")
        residual = []
    else:
        residual = check_residual_naming(SOURCE_DIRS)
        if residual:
            print(f"  WARNING: {len(residual)} files still contain unexpected PPU naming:")
            for filepath in residual[:10]:
                print(f"    {filepath}")
            if len(residual) > 10:
                print(f"    ... and {len(residual) - 10} more")
        else:
            print("  No unexpected hggc/acrand tokens")

    print(f"\nDone in {time.time() - started_at:.1f}s")
    print("=" * 60)
    print(f"  Bulk replace:  {files_modified}")
    print(f"  Residual:      {len(residual)}")
    print("=" * 60)

    if args.dry_run:
        print("\n  [DRY RUN - no files modified]")
    else:
        print("\n  Next steps:")
        print("  1. Convert ACTLIZE: python3 cudafy.py actlize --version=0.5.0 "
              "<xformers>/third_party/actlize/include")
        print("  2. Build: MAX_JOBS=8 python3 setup.py bdist_wheel")

    return 1 if residual else 0


if __name__ == '__main__':
    sys.exit(main())
