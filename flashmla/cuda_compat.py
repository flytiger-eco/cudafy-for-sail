#!/usr/bin/env python3
"""
FlashMLA PPU-original -> CUDA-Compatible Transformation Script.

Converts a FlashMLA ppu-original source tree into a CUDA-compatible one.
Token conversion is a pure regex/dict replacement, mirroring the flash-attention
scripts; the per-file rewrites, multi-line regexes and __HGGCCC__ guards are the
FlashMLA-specific payload on top of that shared engine.

Usage:
    python cuda_compat.py [TARGET_DIR] [--dry-run] [--verbose]

Run from the FlashMLA root directory, or pass it as TARGET_DIR.
"""

import argparse
import os
import re
import sys
import time

# =============================================================================
# Conversion maps (hggc/PPU -> cuda/CU/NV)
# Sorted longest-first at runtime by build_pattern, so declaration order carries
# no meaning and overlapping tokens resolve by length in a single pass.
# __HGGC_ARCH__ is intentionally absent: the PPU hgcc frontend auto-defines it,
# so the cuda-compatible build keeps the native arch guard as-is.
# =============================================================================

CONVERT_RUNTIME_API = {
    # --- functions ---
    'hggcOccupancyMaxPotentialBlockSize': 'cudaOccupancyMaxPotentialBlockSize',
    'hggcTriggerProgrammaticLaunchCompletion': 'cudaTriggerProgrammaticLaunchCompletion',
    'hggcGridDependencySynchronize': 'cudaGridDependencySynchronize',
    'hggcDeviceGetAttribute': 'cudaDeviceGetAttribute',
    'hggcGetDeviceProperties': 'cudaGetDeviceProperties',
    'hggcGetFuncBySymbol': 'cudaGetFuncBySymbol',
    'hggcFuncSetAttribute': 'cudaFuncSetAttribute',
    'hggcFuncGetAttributes': 'cudaFuncGetAttributes',
    'hggcGetErrorString': 'cudaGetErrorString',
    'hggcGetLastError': 'cudaGetLastError',
    'hggcGetErrorName': 'cudaGetErrorName',
    'hggcPeekAtLastError': 'cudaPeekAtLastError',
    'hggcGetDevice': 'cudaGetDevice',
    'hggcLaunchKernelEx': 'cudaLaunchKernelEx',
    'hggcLaunchKernel': 'cudaLaunchKernel',
    'hggcThreadExchangeStreamCaptureMode': 'cudaThreadExchangeStreamCaptureMode',
    'hggcStreamCreateWithFlags': 'cudaStreamCreateWithFlags',
    'hggcStreamIsCapturing': 'cudaStreamIsCapturing',
    'hggcStreamSynchronize': 'cudaStreamSynchronize',
    'hggcStreamDestroy': 'cudaStreamDestroy',
    'hggcMemcpyAsync': 'cudaMemcpyAsync',
    'hggcMemsetAsync': 'cudaMemsetAsync',
    'hggcDeviceSynchronize': 'cudaDeviceSynchronize',
    'hggcSetDevice': 'cudaSetDevice',
    'hggcMemGetInfo': 'cudaMemGetInfo',
    'hggcMalloc': 'cudaMalloc',
    'hggcFree': 'cudaFree',
    'hggcMemcpy': 'cudaMemcpy',
    'hggcEventElapsedTime': 'cudaEventElapsedTime',
    'hggcEventSynchronize': 'cudaEventSynchronize',
    'hggcEventCreate': 'cudaEventCreate',
    'hggcEventDestroy': 'cudaEventDestroy',
    'hggcEventRecord': 'cudaEventRecord',
    'hgLaunchKernelExAD': 'cuLaunchKernelExAD',
    'hgGetErrorName': 'cuGetErrorName',
    'hgGetErrorString': 'cuGetErrorString',

    # --- types ---
    'hggcLaunchConfig_t': 'cudaLaunchConfig_t',
    'hggcLaunchAttribute': 'cudaLaunchAttribute',
    'hggcStream_t': 'cudaStream_t',
    'hggcFunction_t': 'cudaFunction_t',
    'hggcDataType_t': 'cudaDataType_t',
    'hggcEvent_t': 'cudaEvent_t',
    'hggcFuncAttributes': 'cudaFuncAttributes',
    'hggcFuncAttribute': 'cudaFuncAttribute',
    'hggcMemcpyKind': 'cudaMemcpyKind',
    'hggcStreamCaptureStatus': 'cudaStreamCaptureStatus',
    'hggcStreamCaptureMode': 'cudaStreamCaptureMode',
    'HGlaunchAttributeAD': 'CUlaunchAttributeAD',
    'HGlaunchConfigAD': 'CUlaunchConfigAD',
    'HGfunction': 'CUfunction',
    'HGresult': 'CUresult',

    # --- enums / constants ---
    'hggcLaunchAttributeProgrammaticStreamSerialization': 'cudaLaunchAttributeProgrammaticStreamSerialization',
    'hggcFuncAttributeMaxDynamicSharedMemorySize': 'cudaFuncAttributeMaxDynamicSharedMemorySize',
    'hggcOccupancyMaxActiveBlocksPerMultiprocessor': 'cudaOccupancyMaxActiveBlocksPerMultiprocessor',
    'hggcOccupancyDisableCachingOverride': 'cudaOccupancyDisableCachingOverride',
    'hggcDevAttrMaxSharedMemoryPerMultiprocessor': 'cudaDevAttrMaxSharedMemoryPerMultiprocessor',
    'hggcDevAttrMaxSharedMemoryPerBlockOptin': 'cudaDevAttrMaxSharedMemoryPerBlockOptin',
    'hggcDevAttrMultiProcessorCount': 'cudaDevAttrMultiProcessorCount',
    'hggcDevAttrComputeCapabilityMajor': 'cudaDevAttrComputeCapabilityMajor',
    'hggcDevAttrComputeCapabilityMinor': 'cudaDevAttrComputeCapabilityMinor',
    'hggcStreamCaptureModeRelaxed': 'cudaStreamCaptureModeRelaxed',
    'hggcStreamCaptureStatusNone': 'cudaStreamCaptureStatusNone',
    'hggcStreamNonBlocking': 'cudaStreamNonBlocking',
    'hggcStreamDefault': 'cudaStreamDefault',
    'hggcMemcpyDeviceToHost': 'cudaMemcpyDeviceToHost',
    'hggcMemcpyDeviceToDevice': 'cudaMemcpyDeviceToDevice',
    'hggcMemcpyHostToDevice': 'cudaMemcpyHostToDevice',
    'hggcMemcpyHostToHost': 'cudaMemcpyHostToHost',
    'hggcErrorUnknown': 'cudaErrorUnknown',
    'hggcSuccess': 'cudaSuccess',
    'hggcError': 'cudaError',
    'HGGC_SUCCESS': 'CUDA_SUCCESS',
    'HGAD_LAUNCH_ATTRIBUTE_IGNORE': 'CUAD_LAUNCH_ATTRIBUTE_IGNORE',

    # --- device built-in types ---
    '__hg_fp8': '__nv_fp8',
    '__ppu_bfloat162': '__nv_bfloat162',
    '__ppu_bfloat16_raw': '__nv_bfloat16_raw',
    '__ppu_bfloat16': '__nv_bfloat16',
}

CONVERT_HEADER = {
    '<hggc_runtime.h>': '<cuda_runtime.h>',
    '"hggc_runtime.h"': '"cuda_runtime.h"',
    '<hggc_fp16.h>': '<cuda_fp16.h>',
    '<hggc_bf16.h>': '<cuda_bf16.h>',
    '<hggc_pipeline.h>': '<cuda_pipeline.h>',
    '<hggc_awbarrier.h>': '<cuda_awbarrier.h>',
    '<hggc_ad.h>': '"cuda_ad.h"',
    '<hgtx3/hgToolsExt.h>': '<nvtx3/nvToolsExt.h>',
}

# The hgtx (profiling) family plus the RTC compiler macro. __HGGC_ARCH__ and
# __HGGCCC__ are NOT converted: the PPU frontend auto-defines both, and the
# sources use __HGGCCC__ as the platform discriminator for the driver-AD path.
CONVERT_MACRO = {
    '__HGGCCC_RTC__': '__CUDACC_RTC__',
    'HGTX_MESSAGE_TYPE_ASCII': 'NVTX_MESSAGE_TYPE_ASCII',
    'HGTX_VERSION': 'NVTX_VERSION',
    'hgtxEventAttributes_t': 'nvtxEventAttributes_t',
    'hgtxDomainHandle_t': 'nvtxDomainHandle_t',
    'hgtxDomainCreateA': 'nvtxDomainCreateA',
    'hgtxDomainDestroy': 'nvtxDomainDestroy',
    'hgtxDomainRangePushEx': 'nvtxDomainRangePushEx',
    'hgtxDomainRangePop': 'nvtxDomainRangePop',
    'use_hgtx_': 'use_nvtx_',
}

# Diagnostic strings, renamed together with the API they report on, plus the one
# declaration that has to disappear instead of being renamed: api/common.h
# forward-declares hggcStream_t because the PPU SDK headers are not on the host
# compiler's include path, while the compat build gets cudaStream_t from
# cuda_runtime.h via torch -- renaming it would collide with that typedef.
# Longest-first matching makes this block win over the bare hggcStream_t entry.
CONVERT_TEXT = {
    '// Forward-declare hggcStream_t so function signatures match between .cu and .cpp\n'
    'typedef struct HGstream_st* hggcStream_t;\n': '',
    'HG driver error': 'CUDA driver error',
    'HGGC error': 'CUDA error',
}


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
    """Apply dict-based replacement in a single pass."""
    return pattern.sub(lambda m: maps[m.group(0)], content)


def _rewrite_torch_ext_setup_py(content):
    """Transform a setup.py that already uses torch BuildExtension/CUDAExtension.

    Beyond the hgcc identifiers, the compat build also needs the ppu arch flag, the
    cuda driver library and the ptxas resource report. The two insertions are guarded
    so a second run does not duplicate them.
    """
    # --- hgcc -> nvcc identifiers ---
    content = content.replace('hgcc', 'nvcc')

    # --- ppu arch flags -> gencode ---
    content = content.replace(
        'cc_flag.append("-arch=ppu_10")\n'
        'cc_flag.append("-arch=ppu_15")\n',
        'cc_flag.append("-gencode")\n'
        'cc_flag.append("arch=compute_80,code=sm_80")\n')

    # --- cuda driver library, needed by the driver-AD launch path ---
    if "libraries=['cuda']" not in content:
        content = content.replace(
            '        sources=get_sources(),\n'
            '        extra_compile_args={\n',
            '        sources=get_sources(),\n'
            "        libraries=['cuda'],\n"
            '        extra_compile_args={\n')

    # --- ptxas resource report + register usage level ---
    if '--ptxas-options' not in content:
        content = content.replace(
            '                    "--use_fast_math",\n'
            '                    "-mllvm",\n',
            '                    "--use_fast_math",\n'
            '                    "--ptxas-options=-v,--register-usage-level=10",\n'
            '                    "-mllvm",\n')

    return content


def rewrite_setup_py():
    """Transform setup.py to the cuda-compatible CUDAExtension form.

    setup.py is expected to already use torch CUDAExtension, so only the hgcc
    identifiers, the ppu arch flags and the build-tuning args need converting.
    The legacy hand-written HGCCBuildExtension shape is no longer converted.
    """
    targets = ['setup.py']
    count = 0
    for fp in targets:
        if not os.path.isfile(fp):
            continue
        content = open(fp).read()

        if 'HGCCBuildExtension' in content:
            print(f"  WARNING: {fp} uses the legacy HGCCBuildExtension, which this "
                  f"script no longer converts, left untouched")
            continue
        if 'CUDAExtension' not in content:
            print(f"  WARNING: {fp} matches no known shape, left untouched")
            continue

        new_content = _rewrite_torch_ext_setup_py(content)
        if '-arch=ppu_' in new_content or 'hgcc' in new_content:
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
            # Skip the actlize submodule, converted by the actlize script
            if 'actlize' in root:
                continue
            for f in filenames:
                if f.endswith(SUFFIXES):
                    files.append(os.path.join(root, f))
    return files


def transform_content(content, maps, pattern):
    """Apply every conversion to one file's content."""
    # Skip files that carry no ppu-original token
    if not re.search(r'\bhggc|\bPPU10_|__hg_fp8|__ppu_bfloat16|<hgtx3/|<hggc_', content):
        return content

    new_content = replace_content(content, maps, pattern)
    return new_content


def transform_file(filepath, maps, pattern):
    """Apply the conversions to a single file."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
    except (IOError, OSError):
        return False

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
        description='FlashMLA PPU-original -> CUDA-Compatible Transform (no git dependency)')
    parser.add_argument('target_dir', nargs='?', default='.',
                        help='Target FlashMLA root directory')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview without modifying files')
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
    print("FlashMLA PPU-original -> CUDA-Compatible Transform")
    print(f"Target: {target_dir}")
    print("=" * 60)

    t0 = time.time()

    # Step 1: Bulk replacement over the whole source tree (hggc->cuda, hgtx->nvtx,
    # the per-file rewrites and the __HGGCCC__ guards).
    print("\n[1/2] Bulk HGGC/HGTX -> CUDA/NVTX replacement...")
    maps = get_all_convert_maps()
    pattern = build_pattern(maps)

    source_dirs = ['csrc']
    files = find_source_files(source_dirs)
    print(f"  Found {len(files)} source files")

    n_transformed = 0
    for f in sorted(files):
        if args.dry_run:
            content = open(f, encoding='utf-8', errors='replace').read()
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
        print("  [dry-run] Would modify setup.py")
    print(f"  Modified {n_setup} files")

    # Summary
    t1 = time.time()
    print(f"\nDone in {t1 - t0:.1f}s")
    print("=" * 60)
    print(f"  Bulk replace:  {n_transformed}")
    print(f"  Setup.py:      {n_setup}")
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
