#!/usr/bin/env python3
"""
Flash-Attention PPU-original -> CUDA-Compatible Transformation Script.

Converts flash-attention (FA2 + FA3) from ppu-original compilation to nvcc wrapper /
CUDAExtension compilation mode. Pure regex/dict replacement, no git dependency.
Follows acompute's replace_cuda approach.

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
# Conversion maps (hggc/PPU -> cuda/SM)
# Sorted longest-first at runtime to avoid partial matches.
# =============================================================================

CONVERT_RUNTIME_API = {
    'hggcOccupancyMaxActiveBlocksPerMultiprocessor': 'cudaOccupancyMaxActiveBlocksPerMultiprocessor',
    'hggcFuncAttributeMaxDynamicSharedMemorySize': 'cudaFuncAttributeMaxDynamicSharedMemorySize',
    'hggcDevAttrMaxSharedMemoryPerMultiprocessor': 'cudaDevAttrMaxSharedMemoryPerMultiprocessor',
    'hggcDevAttrMaxSharedMemoryPerBlockOptin': 'cudaDevAttrMaxSharedMemoryPerBlockOptin',
    'hggcDevAttrComputeCapabilityMajor': 'cudaDevAttrComputeCapabilityMajor',
    'hggcDevAttrComputeCapabilityMinor': 'cudaDevAttrComputeCapabilityMinor',
    'hggcDevAttrMultiProcessorCount': 'cudaDevAttrMultiProcessorCount',
    'hggcDeviceGetAttribute': 'cudaDeviceGetAttribute',
    'hggcFuncGetAttributes': 'cudaFuncGetAttributes',
    'hggcFuncAttributes': 'cudaFuncAttributes',
    'hggcFuncSetAttribute': 'cudaFuncSetAttribute',
    'hggcGetErrorString': 'cudaGetErrorString',
    'hggcGetLastError': 'cudaGetLastError',
    'hggcGetDevice': 'cudaGetDevice',
    'hggcStream_t': 'cudaStream_t',
    'hggcError_t': 'cudaError_t',
    'hggcSuccess': 'cudaSuccess',
}

CONVERT_HEADER = {
    '<hggc.h>': '<cuda.h>',
    '"hggc.h"': '"cuda.h"',
    '<hggc_runtime.h>': '<cuda_runtime.h>',
    '"hggc_runtime.h"': '"cuda_runtime.h"',
    '<hggc_fp16.h>': '<cuda_fp16.h>',
    '<hggc_bf16.h>': '<cuda_bf16.h>',
    '<hgtx3/hgToolsExt.h>': '<nvtx3/nvToolsExt.h>',
}

CONVERT_MACRO = {
    '__HGGCCC_RTC__': '__CUDACC_RTC__',
    '__HGGCCC__': '__CUDACC__',
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

# Diagnostic strings, renamed together with the API they report on
CONVERT_TEXT = {
    'HGGC error': 'CUDA error',
}

# Applied only on lines containing __CUDA_ARCH__ (after macro replacement)
CONVERT_ARCH_VALUE = {
    '>= 100': '>= 800',
    '== 100': '== 800',
    '== 150': '== 890',
    # PPU1.0 is cuda sm80 and PPU1.5 is sm89; the remaining comparison forms use the
    # same two anchors so an arch check keeps its meaning after the macro swap.
    '<= 100': '<= 800',
    '< 100': '< 800',
    '>= 150': '>= 890',
    # hgcc arch tops out at 150 (== cuda 890); cuda's next gen 900 has no hgcc code
    # yet, so ppu-original spells it '> 150' and the compat form uses '>= 900'.
    '> 150': '>= 900',
    '<= 150': '<= 890',
}

# CU -> SM naming: KernelHardwareInfo carries cu_count in the ppu cutlass and sm_count
# in the cuda-compatible one, so the kernels have to follow the actlize conversion
# (cuda_compat_v1.0.0.py: cu_count -> sm_count). Word boundaries keep compound names
# such as available_cu_count out, they are converted by actlize itself.
CU_TO_SM_RULES = [
    (re.compile(r'\bcu_count\b'), 'sm_count'),
    (re.compile(r'\bCU count\b'), 'SM count'),
]


def get_all_convert_maps():
    """Merge all maps."""
    merged = {}
    merged.update(CONVERT_RUNTIME_API)
    merged.update(CONVERT_HEADER)
    merged.update(CONVERT_MACRO)
    merged.update(CONVERT_TEXT)
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


def apply_convert_arch_values(content):
    """Replace arch values only on lines containing __CUDA_ARCH__."""
    lines = content.split('\n')
    new_lines = []
    for line in lines:
        if '__CUDA_ARCH__' in line:
            for old, new in CONVERT_ARCH_VALUE.items():
                line = line.replace(old, new)
        new_lines.append(line)
    return '\n'.join(new_lines)


def apply_cu_to_sm(content):
    """Rename KernelHardwareInfo cu_count to sm_count, comments and traces included."""
    for pattern, replacement in CU_TO_SM_RULES:
        content = pattern.sub(replacement, content)
    return content


def _rewrite_torch_ext_setup_py(content, is_fa3):
    """Transform a setup.py that already uses torch BuildExtension/CUDAExtension.

    The ppu-original build was migrated off the hand-written HGCCBuildExtension, so the
    only remaining differences from the cuda-compatible form are the extra_compile_args
    key name and the ppu arch flags.
    """
    # --- extra_compile_args key: torch only dispatches on 'cxx' / 'nvcc' ---
    content = content.replace('"hgcc":', '"nvcc":')

    # --- ppu arch flags -> gencode ---
    ppu_arch_flags = (
        '    cc_flag.append("-arch=ppu_10")\n'
        '    cc_flag.append("-arch=ppu_15")\n')
    if is_fa3:
        # FA3 gates sm_90a on DISABLE_SM90; the _write_ninja_file monkey patch in
        # hopper/setup.py then rewrites code=compute_* to code=sm_* for PPU
        content = content.replace(
            ppu_arch_flags,
            '    if not DISABLE_SM90:\n'
            '        cc_flag.append("-gencode")\n'
            '        cc_flag.append("arch=compute_90a,code=sm_90a")\n')
    else:
        content = content.replace(
            ppu_arch_flags,
            '    cc_flag.append("-gencode")\n'
            '    cc_flag.append("arch=compute_80,code=sm_80")\n'
            '    cc_flag.append("-gencode")\n'
            '    cc_flag.append("arch=compute_89,code=sm_89")\n')

    return content


def rewrite_setup_py():
    """Transform setup.py to the cuda-compatible CUDAExtension/BuildExtension form.

    setup.py is expected to already use torch BuildExtension/CUDAExtension, so only
    the extra_compile_args key and the PPU arch flags need converting. The legacy
    hand-written HGCCBuildExtension shape is no longer converted.
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
        if 'hgcc_flags' not in content:
            print(f"  WARNING: {fp} matches no known shape, left untouched")
            continue

        new_content = _rewrite_torch_ext_setup_py(
            content, 'PACKAGE_NAME = "flash_attn_3"' in content)
        if '-arch=ppu_' in new_content or '"hgcc":' in new_content:
            print(f"  WARNING: {fp} still has PPU-only build flags after rewrite")
        if new_content != content:
            open(fp, 'w').write(new_content)
            count += 1
    return count


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
        for root, _, filenames in os.walk(d):
            for f in filenames:
                if f.endswith(SUFFIXES):
                    files.append(os.path.join(root, f))
    return files


def transform_content(content, maps, pattern):
    """Apply every conversion to one file's content."""
    # cu_count is independent of the hggc naming, so it is also renamed in files that
    # are already in compat mode
    new_content = apply_cu_to_sm(content)

    # Skip the hggc replacement if already in compat mode
    if ('cudaStream_t' in new_content and '__CUDA_ARCH__' in new_content
            and 'hggcStream_t' not in new_content):
        return new_content

    new_content = replace_content(new_content, maps, pattern)
    return apply_convert_arch_values(new_content)


def transform_file(filepath, maps, pattern):
    """Apply the conversions to a single file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except (UnicodeDecodeError, UnicodeError):
        return False  # Skip binary or non-UTF-8 files

    new_content = transform_content(content, maps, pattern)
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
    if not os.path.isfile(os.path.join(target_dir, 'setup.py')) or not os.path.isdir(os.path.join(target_dir, 'csrc/flash_attn')):
        print(f"ERROR: {target_dir} is not a flash-attention root directory.")
        sys.exit(1)
    os.chdir(target_dir)

    print("=" * 60)
    print("Flash-Attention PPU-original -> CUDA-Compatible Transform")
    print(f"Target: {target_dir}")
    print("=" * 60)

    t0 = time.time()

    # Step 1: Bulk replacement over the whole source tree (hggc->cuda,
    # hgtx->nvtx, HGGC macros->CUDA macros, cu_count->sm_count, arch values).
    print("\n[1/2] Bulk HGGC/HGTX -> CUDA/NVTX and CU -> SM replacement...")
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
    print(f"  Transformed {n_transformed} files")

    # Step 2: setup.py build-flag transformation
    print("\n[2/2] Setup.py transformation...")
    n_setup = 0
    if not args.dry_run:
        n_setup = rewrite_setup_py()
    else:
        print("  [dry-run] Would modify setup.py and hopper/setup.py")
    print(f"  Modified {n_setup} files")

    # Summary
    t1 = time.time()
    print(f"\nDone in {t1-t0:.1f}s")
    print("=" * 60)
    print(f"  Bulk replace:  {n_transformed}")
    print(f"  Setup.py:      {n_setup}")
    print("=" * 60)

    if args.dry_run:
        print("\n  [DRY RUN - no files modified]")
    else:
        print("\n  Next steps:")
        print("  1. Build FA2: MAX_JOBS=8 python3 setup.py bdist_wheel")
        print("  2. Build FA3: cd hopper && MAX_JOBS=8 python3 setup.py bdist_wheel")


if __name__ == '__main__':
    main()
