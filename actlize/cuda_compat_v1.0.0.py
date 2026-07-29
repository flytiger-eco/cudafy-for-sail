#!/usr/bin/env python3
"""
cuda_compat.py — Convert PPU ACTLIZE v1.0.0 to CUDA-compatible version


Usage:
    python3 cuda_compat_v1.0.0.py <include_dir>

Processing pipeline (5 steps):
    1. Global text replacement (REVERSE_REPLACEMENTS)
       - Runtime API functions (hggc* → cuda*)
       - Runtime types (hggcStream_t → cudaStream_t, etc.)
       - Driver / TMA symbols (HGtensorMap → CUtensorMap, etc.)
       - Device built-in types (__ppu_bfloat16 → __nv_bfloat16, etc.)
       - System header includes (<hggc_*.h> → <cuda_*.h>, etc.)
       - namespace (hggc:: → cuda::)
       - Compiler macros (HGGCCC → CUDACC, etc.)
       - Note: __HGGC_ARCH__ is NOT replaced; original PPU arch numbers (100/150) are preserved
    2. File renames (none — all handled by thin wrappers)
    3. PPU alias injection (PPU_ALIAS_INJECTIONS)
       - Inject using/define aliases into PPU source files so callers using SM* names still compile
       - Covers: arch.h, copy_ppu.hpp, mma_ppu.hpp, epilogue/, gemm/, etc.
    4. Thin wrapper generation (WRAPPER_SPECS)
       - Create pure redirect thin wrappers (#include "xxx_ppu.h") for renamed headers
       - Covers: cute/arch, cute/atom, cutlass/arch, cutlass/epilogue, cutlass/gemm
    5. Architecture macro mapping (create_arch_map_header + inject_arch_map_includes)
       - Generate hggc_arch_map.h: define __HGGC_ARCH__ from __CUDA_ARCH__ (100↔800, 150↔890)
       - Inject #include of the mapping header into all files that use __HGGC_ARCH__
"""

import sys
import os
import re

# ============================================================
# Reverse conversion rules (PPU/HGGC → CUDA); order matters: longer patterns before shorter ones
# ============================================================
REVERSE_REPLACEMENTS = [
    # === Category 5: Device built-in types (longer patterns first) ===
    (r'\b__ppu_bfloat16_raw\b', '__nv_bfloat16_raw'),
    (r'\b__ppu_bfloat162\b', '__nv_bfloat162'),
    (r'\bto_ppu_bfloat16\b', 'to_nv_bfloat16'),
    (r'\b__ppu_bfloat16\b', '__nv_bfloat16'),
    (r'\b__ppu_fp8_e4m3\b', '__nv_fp8_e4m3'),
    (r'\b__ppu_fp8_e5m2\b', '__nv_fp8_e5m2'),
    (r'\b__hg_fp8_e4m3\b', '__nv_fp8_e4m3'),
    (r'\b__hg_fp8_e5m2\b', '__nv_fp8_e5m2'),
    (r'\b__hg_fp8_storage_t\b', '__nv_fp8_storage_t'),

    # === Category 4: CUDA Driver / TMA symbols ===
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

    # === Category 1: CUDA Runtime API functions (longer patterns first) ===
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

    # === Category 2: CUDA Runtime types ===
    (r'\bhggcStreamNonBlocking\b', 'cudaStreamNonBlocking'),
    (r'\bhggcStreamDefault\b', 'cudaStreamDefault'),
    (r'\bhggcStream_t\b', 'cudaStream_t'),
    (r'\bhggcSuccess\b', 'cudaSuccess'),
    (r'\bhggcError_t\b', 'cudaError_t'),

    # === Category 6: System header includes (hggc → cuda; omit <> to handle both quote styles) ===
    (r'\bhggc_runtime_api\.h\b', 'cuda_runtime_api.h'),
    (r'\bhggc_runtime\.h\b', 'cuda_runtime.h'),
    (r'\bhggc_fp16\.h\b', 'cuda_fp16.h'),
    (r'\bhggc_fp8\.h\b', 'cuda_fp8.h'),
    (r'\bhggc_bf16\.h\b', 'cuda_bf16.h'),
    (r'\bacComplex\.h\b', 'cuComplex.h'),
    (r'\bhgComplex\.h\b', 'cuComplex.h'),
    (r'\bacrand_kernel\.h\b', 'curand_kernel.h'),

    # === ac* complex type → cu* reverse mapping (long patterns first) ===
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

    # === M4: CUTLASS internal macros + compiler macro reverse mapping ===
    # Host adapter: type rename → alias injection, file rename → thin wrapper, param/local → no change

    # Driver macros + types
    (r'\bHGlaunchAttribute\b', 'CUlaunchAttribute'),
    (r'\bhgTensorMapEncodeIm2col\b', 'cuTensorMapEncodeIm2col'),

    # Barrier + Clang (internal macros, not converted)

    # Platform macros
    (r'\b__HGGC_STD_MAX\b', '__NV_STD_MAX'),
    (r'\b__HGGC_STD_MIN\b', '__NV_STD_MIN'),


    # CUDA API constants
    (r'\bHGGC_SUCCESS\b', 'CUDA_SUCCESS'),
    (r'\bHGGC_ERROR_UNKNOWN\b', 'CUDA_ERROR_UNKNOWN'),
    (r'\bhggcErrorUnknown\b', 'cudaErrorUnknown'),

    # Compiler macros (HGGCCC → CUDACC, longer patterns first)
    (r'\b__HGGCCC_RTC__\b', '__CUDACC_RTC__'),
    (r'\b__HGGCCC_VER_MAJOR__\b', '__CUDACC_VER_MAJOR__'),
    (r'\b__HGGCCC_VER_MINOR__\b', '__CUDACC_VER_MINOR__'),
    (r'\b__HGGCCC_VERSION__\b', '__CUDACC_VERSION__'),
    (r'\b__HGGCCC__\b', '__CUDACC__'),

    # Architecture macro — do NOT replace __HGGC_ARCH__; preserve original PPU arch numbers
    # create_arch_map_header generates a mapping header to define __HGGC_ARCH__ from __CUDA_ARCH__

    # Clang CUDA macro
    (r'\b__HGGC__\b', '__CUDA__'),

    # === namespace (hggc:: → cuda::) ===
    (r'\bhggc::', 'cuda::'),

    # === SM→CU dimension reverse mappings (PPU cu → CUDA sm, long patterns first) ===
    # Struct member: KernelHardwareInfo::cu_count → sm_count
    (r'\bcu_count\b', 'sm_count'),
    # Static members: GemmUniversalBase::device_cus_ / cu_occupancy_
    (r'\bdevice_cus_\b', 'device_sms_'),
    (r'\bcu_occupancy_\b', 'sm_occupancy_'),
    # Local variables / params
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

    # === NV→PTG dimension reverse mappings (PPG ptg → CUDA nv) ===
    (r'\bPtgType\b', 'NvType'),
    (r'\bPtgTypeV2\b', 'NvTypeV2'),
]

def apply_reverse_replacements(content):
    """Apply reverse conversion rules (PPU/HGGC → CUDA) to file content."""
    for pattern, replacement in REVERSE_REPLACEMENTS:
        content = re.sub(pattern, replacement, content)
    return content


def process_directory(include_dir):
    """Walk the include/ directory and apply reverse conversion to all C/C++ files."""
    changed_files = []
    total_files = 0

    # actlize/include dir
    for root, dirs, files in os.walk(include_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            total_files += 1

            # Process only text files (C/C++ header/source)
            if not any(fname.endswith(ext) for ext in
                       ('.h', '.hpp', '.cuh', '.cu', '.c', '.cpp', '.inl', '.inc')):
                continue

            try:
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                    original = f.read()
            except Exception:
                continue

            modified = apply_reverse_replacements(original)

            if modified != original:
                with open(fpath, 'w', encoding='utf-8') as f:
                    f.write(modified)
                rel = os.path.relpath(fpath, include_dir)
                changed_files.append(rel)

    # Also process ../tools/util/include relative to include_dir
    tools_util_include = os.path.normpath(os.path.join(include_dir, '..', 'tools', 'util', 'include'))
    if os.path.isdir(tools_util_include):
        for root, dirs, files in os.walk(tools_util_include):
            for fname in files:
                fpath = os.path.join(root, fname)
                total_files += 1

                # Process only text files (C/C++ header/source)
                if not any(fname.endswith(ext) for ext in
                           ('.h', '.hpp', '.cuh', '.cu', '.c', '.cpp', '.inl', '.inc')):
                    continue

                try:
                    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                        original = f.read()
                except Exception:
                    continue

                modified = apply_reverse_replacements(original)

                if modified != original:
                    with open(fpath, 'w', encoding='utf-8') as f:
                        f.write(modified)
                    rel = os.path.relpath(fpath, tools_util_include)
                    changed_files.append(f'tools/util/include/{rel}')

    return changed_files, total_files


# ============================================================
# PPU alias injection + thin wrapper file mapping table
# ============================================================

# PPU source file alias injection map: (namespace, alias_body)
# namespace: the C++ namespace for the aliases; the function wraps them in a namespace block
# alias_body: alias contents (using/define etc., without the namespace block)
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


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 cuda_compat.py <include_dir>")
        sys.exit(1)

    include_dir = sys.argv[1]
    if not os.path.isdir(include_dir):
        print(f"Error: directory not found: {include_dir}")
        sys.exit(1)

    print(f"[cuda_compat] include dir: {include_dir}")
    print("[cuda_compat] reverse conversion + PPU alias injection + thin wrapper generation + arch macro mapping")

    changed, total = process_directory(include_dir)

    # Handle file renames (none — all handled by wrappers)
    renames = []
    for src_rel, dst_rel in renames:
        src = os.path.join(include_dir, src_rel)
        dst = os.path.join(include_dir, dst_rel)
        if os.path.exists(src) and not os.path.exists(dst):
            os.rename(src, dst)
            changed.append(f"{src_rel} -> {dst_rel} (rename)")

    # Inject aliases into PPU source files + generate thin wrappers + arch macro mapping
    inject_ppu_aliases(include_dir, changed)
    create_wrappers(include_dir, changed)
    create_arch_map_header(include_dir, changed)
    inject_arch_map_includes(include_dir, changed)

    print(f"[cuda_compat] files scanned: {total}")
    print(f"[cuda_compat] files changed: {len(changed)}")
    for f in sorted(changed):
        print(f"  {f}")


if __name__ == "__main__":
    main()
