#!/usr/bin/env python3
"""
Flash-Attention PPU-original → CUDA-Compatible Transformation Script (v2.7.4).

Converts flash-attention (FA2) from ppu-original compilation to nvcc wrapper /
CUDAExtension compilation. Pure text-based conversion, no git dependency.

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
# Conversion maps (hggc/PPU → cuda/SM)
# =============================================================================

CONVERT_RUNTIME_API = {
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

CONVERT_HEADER = {
    '<hggc.h>': '<cuda.h>',
    '"hggc.h"': '"cuda.h"',
    '<hggc_fp16.h>': '<cuda_fp16.h>',
    '<hggc_bf16.h>': '<cuda_bf16.h>',
    '<hggc_runtime.h>': '<cuda_runtime.h>',
    '"hggc_runtime.h"': '"cuda_runtime.h"',
    '<hgtx3/hgToolsExt.h>': '<nvtx3/nvToolsExt.h>',
}

CONVERT_MACRO = {
    '__HGGCCC_RTC__': '__CUDACC_RTC__',
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

# Diagnostic strings, renamed together with the API they report on
CONVERT_TEXT = {
    'HGGC error': 'CUDA error',
}


def get_all_convert_maps():
    """Merge all conversion maps, sorted longest-first."""
    merged = {}
    merged.update(CONVERT_RUNTIME_API)
    merged.update(CONVERT_HEADER)
    merged.update(CONVERT_MACRO)
    merged.update(CONVERT_TEXT)
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


# =============================================================================
# setup.py transformation (pure text, no git)
# =============================================================================

def _rewrite_torch_ext_setup_py(content, is_fa3):
    """Transform a setup.py that already uses torch BuildExtension/CUDAExtension.

    The ppu-original build was migrated off the hand-written HGCCBuildExtension, so the
    only remaining differences from the cuda-compatible form are the
    extra_compile_args key name and the ppu arch flags.
    """
    # --- extra_compile_args key: torch only dispatches on 'cxx' / 'nvcc' ---
    content = content.replace('"hgcc":', '"nvcc":')

    # --- ppu arch flags → gencode ---
    ppu_arch_flags = (
        '    cc_flag.append("-arch=ppu_10")\n'
        '    cc_flag.append("-arch=ppu_15")\n')
    if is_fa3:
        # DISABLE_SM90 is hardcoded True in this PPU-only FA3 build (SM90 removed), so
        # the cuda-compatible build emits no -gencode at all and cc_flag goes away
        content = content.replace('    cc_flag = []\n' + ppu_arch_flags + '\n', '')
        content = content.replace('] + feature_args + cc_flag,', '] + feature_args,')
    else:
        content = content.replace(
            ppu_arch_flags,
            '    if "80" in cuda_archs():\n'
            '        cc_flag.append("-gencode")\n'
            '        cc_flag.append("arch=compute_80,code=sm_80")\n'
            '    if CUDA_HOME is not None:\n'
            '        _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)\n'
            '        if bare_metal_version >= Version("11.8") and "90" in cuda_archs():\n'
            '            cc_flag.append("-gencode")\n'
            '            cc_flag.append("arch=compute_90,code=sm_90")\n'
            '        if bare_metal_version >= Version("12.8") and "100" in cuda_archs():\n'
            '            cc_flag.append("-gencode")\n'
            '            cc_flag.append("arch=compute_100,code=sm_100")\n'
            '        if bare_metal_version >= Version("12.8") and "120" in cuda_archs():\n'
            '            cc_flag.append("-gencode")\n'
            '            cc_flag.append("arch=compute_120,code=sm_120")\n')

    return content


def transform_setup_py():
    """Transform setup.py to the cuda-compatible CUDAExtension/BuildExtension form.

    Both setup.py files are expected to already use torch BuildExtension/CUDAExtension,
    so only the extra_compile_args key and the PPU arch flags need converting. The
    legacy hand-written HGCCBuildExtension shape is no longer converted.
    """
    targets = ['setup.py', 'hopper/setup.py']
    count = 0
    for fp in targets:
        if not os.path.isfile(fp):
            continue
        content = open(fp).read()

        if 'HGCCBuildExtension' in content:
            print(f"  WARNING: {fp} uses the legacy HGCCBuildExtension, which this "
                  f"script no longer converts, left untouched")
            continue
        if '-arch=ppu_' not in content:
            print(f"  WARNING: {fp} matches no known shape, left untouched")
            continue

        new_content = _rewrite_torch_ext_setup_py(content, fp.startswith('hopper/'))
        if '-arch=ppu_' in new_content or '"hgcc":' in new_content:
            print(f"  WARNING: {fp} still has PPU-only build flags after rewrite")
        if new_content != content:
            open(fp, 'w').write(new_content)
            count += 1
    return count


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


def transform_content(content, maps, pattern):
    """Apply every conversion to one file's content."""
    # Skip the hggc replacement if already in compat mode
    if ('cudaStream_t' in content and '__CUDA_ARCH__' in content
            and 'hggcStream_t' not in content):
        return content

    new_content = replace_content(content, maps, pattern)
    return new_content


def transform_file(filepath, maps, pattern):
    """Apply the conversions to a single file."""
    with open(filepath, 'r') as f:
        content = f.read()

    new_content = transform_content(content, maps, pattern)
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

    # Step 1: Bulk replacement (includes CU -> SM renaming)
    print("\n[1/2] Bulk HGGC → CUDA and CU → SM replacement...")
    maps = get_all_convert_maps()
    pattern = build_pattern(maps)

    source_dirs = ['csrc/flash_attn', 'hopper']
    files = find_source_files(source_dirs)
    print(f"  Found {len(files)} source files")

    n_transformed = 0
    for f in files:
        if args.dry_run:
            content = open(f).read()
            if transform_content(content, maps, pattern) != content:
                n_transformed += 1
                if args.verbose:
                    print(f"    [dry-run] {f}")
        else:
            if transform_file(f, maps, pattern):
                n_transformed += 1
                if args.verbose:
                    print(f"    {f}")
    print(f"  Transformed: {n_transformed} files")

    # Step 2: setup.py transformation
    print("\n[2/2] Setup.py transformation...")
    n_setup = 0
    if not args.dry_run:
        n_setup = transform_setup_py()
    else:
        print("  [dry-run] Would rewrite setup.py and hopper/setup.py for CUDAExtension")
    print(f"  Modified: {n_setup} files")

    # Summary
    t1 = time.time()
    print(f"\n{'=' * 60}")
    print(f"Done in {t1-t0:.1f}s")
    print(f"  Bulk replace:   {n_transformed}")
    print(f"  Setup.py:       {n_setup}")
    print(f"{'=' * 60}")

    if args.dry_run:
        print("\n  [DRY RUN - no files modified]")
    else:
        print("\n  Next: python setup.py bdist_wheel")


if __name__ == '__main__':
    main()
