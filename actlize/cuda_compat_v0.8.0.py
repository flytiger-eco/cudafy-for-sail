#!/usr/bin/env python3
"""ACTLIZE v0.8.0 backward compatibility aliases and symbol replacement.

This script provides four functionalities:
1. apply(): Append backward compatibility aliases (using SM_xxx = PPU_xxx) to source files.
2. replace_symbols(): Replace HGCC/PPU symbol names back to CUDA/CU names in source files.
   Covers compiler macros, API identifiers, header paths and header prefixes.
3. create_file_shims(): Create thin shim headers at old file names (cuda_*) that
   #include the new renamed files (hggc_*), providing backward compatibility.
4. inject_arch_header(): Generate cute/hggc_arch_compat.h dynamically and
   inject #include into files that reference __HGGC_ARCH__, providing
   CUDA_ARCH → HGGC_ARCH mapping.

Usage:
  python3 cuda_compat_v0.8.0.py <include_dir>
"""
import os
import re
import sys

# Each entry: (file_path, namespace, list_of_alias_lines)
# file_path is relative to the <path> argument (typically the include/ directory).
ALIASES = [
    # ========================================================================
    # cutlass::arch — arch tag (SM70, SM72, SM75, SM80, SM86, SM90)
    # ========================================================================
    ("cutlass/arch/arch.h", "cutlass::arch", [
        "using Sm50 = PPU0010;",
        "using Sm60 = PPU0010;",
        "using Sm61 = PPU0010;",
        "using Sm70 = PPU0010;",
        "using Sm72 = PPU0010;",
        "using Sm75 = PPU0010;",
        "using Sm80 = PPU0010;",
        "using Sm86 = PPU0010;",
        "using Sm90 = PPU0015;",
        # Backward compatibility: SmId() wraps CuId() for old API consumers
        # Guarded + CUTLASS_DEVICE so host compilation doesn't see a __device__ call
        "#if defined(__CUDACC__) || defined(__CUDACC_RTC__) || (defined(__clang__) && defined(__CUDA__))",
        "CUTLASS_DEVICE",
        "int SmId() { return CuId(); }",
        "#endif",
    ]),

    # ========================================================================
    # cute — MMA operation structs (SM61/70/75/80 merged into mma_ppu, 74)
    # ========================================================================
    ("cute/arch/mma_ppu.hpp", "cute", [
        # --- F16 ---
        "using SM70_8x8x4_F16F16F16F16_TN     = PPU_8x8x4_F16F16F16F16_TN;",
        "using SM70_8x8x4_F16F16F16F16_NT     = PPU_8x8x4_F16F16F16F16_NT;",
        "using SM70_8x8x4_F16F16F16F16_NN     = PPU_8x8x4_F16F16F16F16_NN;",
        "using SM70_8x8x4_F16F16F16F16_TT     = PPU_8x8x4_F16F16F16F16_TT;",
        # --- F32 x F16 ---
        "using SM70_8x8x4_F32F16F16F32_TN     = PPU_8x8x4_F32F16F16F32_TN;",
        "using SM70_8x8x4_F32F16F16F32_NT     = PPU_8x8x4_F32F16F16F32_NT;",
        "using SM70_8x8x4_F32F16F16F32_NN     = PPU_8x8x4_F32F16F16F32_NN;",
        "using SM70_8x8x4_F32F16F16F32_TT     = PPU_8x8x4_F32F16F16F32_TT;",
        # --- SM75 ---
        "using SM75_16x8x8_F32F16F16F32_TN = PPU_16x8x8_F32F16F16F32_TN;",
        "using SM75_8x8x16_S32S8S8S32_TN  = PPU_8x8x16_S32S8S8S32_TN;",
        # --- F16 ---
        "using SM80_16x8x8_F16F16F16F16_TN     = PPU_16x8x8_F16F16F16F16_TN;",
        "using SM80_16x8x16_F16F16F16F16_TN    = PPU_16x8x16_F16F16F16F16_TN;",
        # --- F32 x F16 ---
        "using SM80_16x8x8_F32F16F16F32_TN     = PPU_16x8x8_F32F16F16F32_TNV2;",
        "using SM80_16x8x16_F32F16F16F32_TN    = PPU_16x8x16_F32F16F16F32_TN;",
        # --- BF16 ---
        "using SM80_16x8x8_F32BF16BF16F32_TN   = PPU_16x8x8_F32BF16BF16F32_TN;",
        "using SM80_16x8x16_F32BF16BF16F32_TN  = PPU_16x8x16_F32BF16BF16F32_TN;",
        # --- TF32 ---
        "using SM80_16x8x4_F32TF32TF32F32_TN   = PPU_16x8x4_F32TF32TF32F32_TN;",
        "using SM80_16x8x8_F32TF32TF32F32_TN   = PPU_16x8x8_F32TF32TF32F32_TN;",
        # --- S8 x S8 ---
        "using SM80_8x8x16_S32S8S8S32_TN       = PPU_8x8x16_S32S8S8S32_TN;",
        "using SM80_8x8x16_S32S8S8S32_TN_SATURATE       = PPU_8x8x16_S32S8S8S32_TN_SATURATE;",
        "using SM80_16x8x16_S32S8S8S32_TN      = PPU_16x8x16_S32S8S8S32_TN;",
        "using SM80_16x8x16_S32S8S8S32_TN_SATURATE      = PPU_16x8x16_S32S8S8S32_TN_SATURATE;",
        "using SM80_16x8x32_S32S8S8S32_TN      = PPU_16x8x32_S32S8S8S32_TN;",
        "using SM80_16x8x32_S32S8S8S32_TN_SATURATE      = PPU_16x8x32_S32S8S8S32_TN_SATURATE;",
        # --- S8 x U8 ---
        "using SM80_8x8x16_S32S8U8S32_TN       = PPU_8x8x16_S32S8U8S32_TN;",
        "using SM80_8x8x16_S32S8U8S32_TN_SATURATE       = PPU_8x8x16_S32S8U8S32_TN_SATURATE;",
        "using SM80_16x8x16_S32S8U8S32_TN      = PPU_16x8x16_S32S8U8S32_TN;",
        "using SM80_16x8x16_S32S8U8S32_TN_SATURATE      = PPU_16x8x16_S32S8U8S32_TN_SATURATE;",
        "using SM80_16x8x32_S32S8U8S32_TN      = PPU_16x8x32_S32S8U8S32_TN;",
        "using SM80_16x8x32_S32S8U8S32_TN_SATURATE      = PPU_16x8x32_S32S8U8S32_TN_SATURATE;",
        # --- U8 x S8 ---
        "using SM80_8x8x16_S32U8S8S32_TN       = PPU_8x8x16_S32U8S8S32_TN;",
        "using SM80_8x8x16_S32U8S8S32_TN_SATURATE       = PPU_8x8x16_S32U8S8S32_TN_SATURATE;",
        "using SM80_16x8x16_S32U8S8S32_TN      = PPU_16x8x16_S32U8S8S32_TN;",
        "using SM80_16x8x16_S32U8S8S32_TN_SATURATE      = PPU_16x8x16_S32U8S8S32_TN_SATURATE;",
        "using SM80_16x8x32_S32U8S8S32_TN      = PPU_16x8x32_S32U8S8S32_TN;",
        "using SM80_16x8x32_S32U8S8S32_TN_SATURATE      = PPU_16x8x32_S32U8S8S32_TN_SATURATE;",
        # --- U8 x U8 ---
        "using SM80_8x8x16_S32U8U8S32_TN       = PPU_8x8x16_S32U8U8S32_TN;",
        "using SM80_8x8x16_S32U8U8S32_TN_SATURATE       = PPU_8x8x16_S32U8U8S32_TN_SATURATE;",
        "using SM80_16x8x16_S32U8U8S32_TN      = PPU_16x8x16_S32U8U8S32_TN;",
        "using SM80_16x8x16_S32U8U8S32_TN_SATURATE      = PPU_16x8x16_S32U8U8S32_TN_SATURATE;",
        "using SM80_16x8x32_S32U8U8S32_TN      = PPU_16x8x32_S32U8U8S32_TN;",
        "using SM80_16x8x32_S32U8U8S32_TN_SATURATE      = PPU_16x8x32_S32U8U8S32_TN_SATURATE;",
        # --- S4 ---
        "using SM80_8x8x32_S32S4S4S32_TN       = PPU_8x8x32_S32S4S4S32_TN;",
        "using SM80_8x8x32_S32S4S4S32_TN_SATURATE       = PPU_8x8x32_S32S4S4S32_TN_SATURATE;",
        "using SM80_16x8x32_S32S4S4S32_TN      = PPU_16x8x32_S32S4S4S32_TN;",
        "using SM80_16x8x32_S32S4S4S32_TN_SATURATE      = PPU_16x8x32_S32S4S4S32_TN_SATURATE;",
        "using SM80_16x8x64_S32S4S4S32_TN      = PPU_16x8x64_S32S4S4S32_TN;",
        "using SM80_16x8x64_S32S4S4S32_TN_SATURATE      = PPU_16x8x64_S32S4S4S32_TN_SATURATE;",
        # --- S4 x U4 ---
        "using SM80_8x8x32_S32S4U4S32_TN       = PPU_8x8x32_S32S4U4S32_TN;",
        "using SM80_8x8x32_S32S4U4S32_TN_SATURATE       = PPU_8x8x32_S32S4U4S32_TN_SATURATE;",
        "using SM80_16x8x32_S32S4U4S32_TN      = PPU_16x8x32_S32S4U4S32_TN;",
        "using SM80_16x8x32_S32S4U4S32_TN_SATURATE      = PPU_16x8x32_S32S4U4S32_TN_SATURATE;",
        "using SM80_16x8x64_S32S4U4S32_TN      = PPU_16x8x64_S32S4U4S32_TN;",
        "using SM80_16x8x64_S32S4U4S32_TN_SATURATE      = PPU_16x8x64_S32S4U4S32_TN_SATURATE;",
        # --- U4 x S4 ---
        "using SM80_8x8x32_S32U4S4S32_TN       = PPU_8x8x32_S32U4S4S32_TN;",
        "using SM80_8x8x32_S32U4S4S32_TN_SATURATE       = PPU_8x8x32_S32U4S4S32_TN_SATURATE;",
        "using SM80_16x8x32_S32U4S4S32_TN      = PPU_16x8x32_S32U4S4S32_TN;",
        "using SM80_16x8x32_S32U4S4S32_TN_SATURATE      = PPU_16x8x32_S32U4S4S32_TN_SATURATE;",
        "using SM80_16x8x64_S32U4S4S32_TN      = PPU_16x8x64_S32U4S4S32_TN;",
        "using SM80_16x8x64_S32U4S4S32_TN_SATURATE      = PPU_16x8x64_S32U4S4S32_TN_SATURATE;",
        # --- U4 x U4 ---
        "using SM80_8x8x32_S32U4U4S32_TN       = PPU_8x8x32_S32U4U4S32_TN;",
        "using SM80_8x8x32_S32U4U4S32_TN_SATURATE       = PPU_8x8x32_S32U4U4S32_TN_SATURATE;",
        "using SM80_16x8x32_S32U4U4S32_TN      = PPU_16x8x32_S32U4U4S32_TN;",
        "using SM80_16x8x32_S32U4U4S32_TN_SATURATE      = PPU_16x8x32_S32U4U4S32_TN_SATURATE;",
        "using SM80_16x8x64_S32U4U4S32_TN      = PPU_16x8x64_S32U4U4S32_TN;",
        "using SM80_16x8x64_S32U4U4S32_TN_SATURATE      = PPU_16x8x64_S32U4U4S32_TN_SATURATE;",
        # --- U1 (XORPOPC) ---
        "using SM80_8x8x128_S32U1U1S32_TN_XORPOPC   = PPU_8x8x128_S32U1U1S32_TN_XORPOPC;",
        "using SM80_16x8x128_S32U1U1S32_TN_XORPOPC  = PPU_16x8x128_S32U1U1S32_TN_XORPOPC;",
        "using SM80_16x8x256_S32U1U1S32_TN_XORPOPC  = PPU_16x8x256_S32U1U1S32_TN_XORPOPC;",
        # --- SM61 ---
        "using SM61_DP4A = PPU_DP4A;",
    ]),

    # ========================================================================
    # cute — MMA Traits layout aliases (5)
    # Note: MMA_Traits<SM70_...> specializations are NOT needed here because
    # the source file already defines MMA_Traits<PPU_...>, and the using
    # aliases (SM70_... = PPU_...) make them equivalent. Adding explicit
    # specializations would cause "redefinition" errors.
    # ========================================================================
    ("cute/atom/mma_traits_ppu.hpp", "cute", [
        # --- Layout aliases ---
        "using SM70_QuadPair = PPU_QuadPair;",
        "using SM70_8x4_Row  = PPU_8x4_Row;",
        "using SM70_8x4_Col  = PPU_8x4_Col;",
        "using SM70_8x8_16b  = PPU_8x8_16b;",
        "using SM70_8x8_32b  = PPU_8x8_32b;",
    ]),

    # ========================================================================
    # cutlass::gemm — Dispatch policy structs (SM70 + SM80)
    # ========================================================================
    ("cutlass/gemm/dispatch_policy.hpp", "cutlass::gemm", [
        "using MainloopSm70TwoStageUnpredicated = MainloopPPU0010TwoStageUnpredicated;",
        "using MainloopSm70TwoStage = MainloopPPU0010TwoStage;",
        "template<int Stages_>\nusing MainloopSm80CpAsyncUnpredicated = MainloopPPU0010CpAsyncUnpredicated<Stages_>;",
        "template<int Stages_>\nusing MainloopSm80CpAsync = MainloopPPU0010CpAsync<Stages_>;",
    ]),

    # ========================================================================
    # cutlass::gemm — PPU dispatch policy structs (2)
    # ========================================================================
    ("ppu/cutlass/gemm/dispatch_policy.hpp", "cutlass::gemm", [
        "using MainloopSm70TwoStageUnpredicatedLdmatrix = MainloopPPU0010TwoStageUnpredicatedLdmatrix;",
        "using MainloopSm70TwoStageLdmatrix = MainloopPPU0010TwoStageLdmatrix;",
    ]),

    # ========================================================================
    # cute — Copy operation structs (SM75/80 merged into copy_ppu, 10)
    # ========================================================================
    ("cute/arch/copy_ppu.hpp", "cute", [
        "using SM75_U32x1_LDSM_N = PPU_U32x1_LDSM_N;",
        "using SM75_U32x2_LDSM_N = PPU_U32x2_LDSM_N;",
        "using SM75_U32x4_LDSM_N = PPU_U32x4_LDSM_N;",
        "using SM75_U16x2_LDSM_T = PPU_U16x2_LDSM_T;",
        "using SM75_U16x4_LDSM_T = PPU_U16x4_LDSM_T;",
        "using SM75_U16x8_LDSM_T = PPU_U16x8_LDSM_T;",
        "template <class TS, class TD = TS>\nusing SM80_CP_ASYNC_CACHEALWAYS = PPU_CP_ASYNC_CACHEALWAYS<TS, TD>;",
        "template <class TS, class TD = TS>\nusing SM80_CP_ASYNC_CACHEGLOBAL = PPU_CP_ASYNC_CACHEGLOBAL<TS, TD>;",
        "template <class TS, class TD = TS>\nusing SM80_CP_ASYNC_CACHEALWAYS_ZFILL = PPU_CP_ASYNC_CACHEALWAYS_ZFILL<TS, TD>;",
        "template <class TS, class TD = TS>\nusing SM80_CP_ASYNC_CACHEGLOBAL_ZFILL = PPU_CP_ASYNC_CACHEGLOBAL_ZFILL<TS, TD>;",
    ]),

    # ========================================================================
    # cutlass::epilogue::threadblock — Visitor template aliases (SM80, 2)
    # ========================================================================
    ("cutlass/epilogue/threadblock/fusion/visitor_2x.hpp", "cutlass::epilogue::threadblock", [
        "template <class NodeOp, class... ChildOps>\nusing Sm80EVT = PPU0010EVT<NodeOp, ChildOps...>;",
        "template <class ElementCompute, class EdgeTuple, class... Ops>\nusing Sm80TopologicalVisitor = PPU0010TopologicalVisitor<ElementCompute, EdgeTuple, Ops...>;",
    ]),

    # ========================================================================
    # SM90 -> PPU aliases
    # ========================================================================
    # cute — copy_sm90 STSM: stmatrix is not supported on PPU 1.0/1.5;

    # cute — copy_sm90_tma: TMA arch instructions not supported on PPU;

    # cute — mma_sm90: F64/C64 MMA not supported on PPU 1.0/1.5;
    # F64/C64 types removed from cute/arch/mma_ppu.hpp. No SM90 F64/C64 MMA aliases.

    # cutlass::epilogue — dispatch_policy (2)
    ("cutlass/epilogue/dispatch_policy.hpp", "cutlass::epilogue", [
        "template<int StagesC_, int StagesD_, int FragmentSize_, bool ReuseSmemC_>\nusing Sm90TmaWarpSpecialized = PPU0015TmaWarpSpecialized<StagesC_, StagesD_, FragmentSize_, ReuseSmemC_>;",
        "template<int StagesC_, int StagesD_, int FragmentSize_ = 2>\nusing Sm90TmaWarpSpecializedBiasElementwise = PPU0015TmaWarpSpecializedBiasElementwise<StagesC_, StagesD_, FragmentSize_>;",
    ]),

    # cutlass::gemm::kernel::detail — tile_scheduler_params (3)
    ("cutlass/gemm/kernel/tile_scheduler_params.h", "cutlass::gemm::kernel::detail", [
        "using PersistentTileSchedulerSm90Params = PersistentTileSchedulerParams;",
        "using PersistentTileSchedulerSm90StreamKParams = PersistentTileSchedulerPPUStreamKParams;",
        "template<class ProblemShape>\nusing PersistentTileSchedulerSm90GroupParams = PersistentTileSchedulerPPUGroupParams<ProblemShape>;",
    ]),

    # cutlass::gemm::kernel::detail — ppu_tile_scheduler (1)
    ("cutlass/gemm/kernel/persistent_tile_scheduler.hpp", "cutlass::gemm::kernel::detail", [
        "using PersistentTileSchedulerSm90 = PersistentTileScheduler;",
    ]),

    # cutlass::gemm::kernel::detail — ppu_tile_scheduler_stream_k (1)
    ("cutlass/gemm/kernel/tile_scheduler_stream_k.hpp", "cutlass::gemm::kernel::detail", [
        "template<class TileShape, class ClusterShape>\nusing PersistentTileSchedulerSm90StreamK = PersistentTileSchedulerPPUStreamK<TileShape, ClusterShape>;",
    ]),

    # cutlass::gemm::kernel::detail — ppu_tile_scheduler_group (1)
    ("cutlass/gemm/kernel/tile_scheduler_group.hpp", "cutlass::gemm::kernel::detail", [
        "template<class GroupProblemShape>\nusing PersistentTileSchedulerSm90Group = PersistentTileSchedulerPPUGroup<GroupProblemShape>;",
    ]),

    # cutlass::epilogue::fusion — ppu_callbacks (13): simplified aliases using canonical PPU0015 names
    ("cutlass/epilogue/fusion/ppu_callbacks_tma_warpspecialized.hpp", "cutlass::epilogue::fusion", [
        "template<class NodeOp, class... ChildOps>\nusing Sm90EVT = PPU0015EVT<NodeOp, ChildOps...>;",
        "template<class ElementOutput, class ElementCompute, class ElementSource = ElementOutput, class ElementScalar = ElementCompute, FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest>\nusing Sm90LinearCombination = PPU0015LinearCombination<ElementOutput, ElementCompute, ElementSource, ElementScalar, RoundStyle>;",
        "template<template<class> class ActivationFn, class ElementOutput, class ElementCompute, class ElementSource = ElementOutput, class ElementScalar = ElementCompute, FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest>\nusing Sm90LinCombEltAct = PPU0015LinCombEltAct<ActivationFn, ElementOutput, ElementCompute, ElementSource, ElementScalar, RoundStyle>;",
        "template<class CtaTileShapeMNK, class ElementOutput, class ElementCompute, class ElementBias = ElementOutput, class ElementSource = ElementOutput, class ElementScalar = ElementCompute, int AlignmentBias = 128 / sizeof_bits_v<ElementBias>, FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest>\nusing Sm90LinCombPerRowBias = PPU0015LinCombPerRowBias<CtaTileShapeMNK, ElementOutput, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle>;",
        "template<class CtaTileShapeMNK, class ElementOutput, class ElementCompute, class ElementBias = ElementOutput, class ElementSource = ElementOutput, class ElementScalar = ElementCompute, int AlignmentBias = 128 / sizeof_bits_v<ElementBias>, int AlignmentScalar = 128 / sizeof_bits_v<ElementScalar>, FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest>\nusing Sm90PerRowLinCombPerRowBias = PPU0015PerRowLinCombPerRowBias<CtaTileShapeMNK, ElementOutput, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, AlignmentScalar, RoundStyle>;",
        "template<class CtaTileShapeMNK, class ElementOutput, class ElementCompute, class ElementBias = ElementOutput, class ElementSource = ElementOutput, class ElementScalar = ElementCompute, int AlignmentBias = 128 / sizeof_bits_v<ElementBias>, FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest>\nusing Sm90ScaledLinCombPerRowBias = PPU0015ScaledLinCombPerRowBias<CtaTileShapeMNK, ElementOutput, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle>;",
        "template<class CtaTileShapeMNK, template<class> class ActivationFn, class ElementOutput, class ElementCompute, class ElementBias = ElementOutput, class ElementSource = ElementOutput, class ElementScalar = ElementCompute, int AlignmentBias = 128 / sizeof_bits_v<ElementBias>, FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest>\nusing Sm90LinCombPerRowBiasEltAct = PPU0015LinCombPerRowBiasEltAct<CtaTileShapeMNK, ActivationFn, ElementOutput, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle>;",
        "template<class CtaTileShapeMNK, class EpilogueTile, int Stages, class StrideAux, class SmemLayoutAtom, class CopyOpR2S, template<class> class ActivationFn, class ElementOutput, class ElementCompute, class ElementAux = ElementOutput, class ElementBias = ElementOutput, class ElementSource = ElementOutput, class ElementScalar = ElementCompute, int AlignmentAux = 128 / sizeof_bits_v<ElementAux>, int AlignmentBias = 128 / sizeof_bits_v<ElementBias>, FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest>\nusing Sm90LinCombPerRowBiasEltActAux = PPU0015LinCombPerRowBiasEltActAux<CtaTileShapeMNK, EpilogueTile, Stages, StrideAux, SmemLayoutAtom, CopyOpR2S, ActivationFn, ElementOutput, ElementCompute, ElementAux, ElementBias, ElementSource, ElementScalar, AlignmentAux, AlignmentBias, RoundStyle>;",
        "template<class CtaTileShapeMNK, template<class> class ActivationFn, class ElementOutput, class ElementCompute, class ElementBias = ElementOutput, class ElementSource = ElementOutput, class ElementScalar = ElementCompute, int AlignmentBias = 128 / sizeof_bits_v<ElementBias>, int AlignmentScalar = 128 / sizeof_bits_v<ElementScalar>, FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest>\nusing Sm90PerRowLinCombPerRowBiasEltAct = PPU0015PerRowLinCombPerRowBiasEltAct<CtaTileShapeMNK, ActivationFn, ElementOutput, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, AlignmentScalar, RoundStyle>;",
        "template<class CtaTileShapeMNK, template<class> class ActivationFn, class ElementOutput, class ElementCompute, class ElementBias = ElementOutput, class ElementSource = ElementOutput, class ElementScalar = ElementCompute, int AlignmentBias = 128 / sizeof_bits_v<ElementBias>, FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest>\nusing Sm90ScaledLinCombPerRowBiasEltAct = PPU0015ScaledLinCombPerRowBiasEltAct<CtaTileShapeMNK, ActivationFn, ElementOutput, ElementCompute, ElementBias, ElementSource, ElementScalar, AlignmentBias, RoundStyle>;",
        "template<class CtaTileShapeMNK, class EpilogueTile, int Stages, class StrideAux, class SmemLayoutAtom, class CopyOpS2R, template<class> class ActivationFn, class ElementOutput, class ElementCompute, class ElementAux = ElementOutput, class ElementSource = ElementOutput, class ElementScalar = ElementCompute, int AlignmentAux = 128 / sizeof_bits_v<ElementAux>, FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest>\nusing Sm90LinCombDeEltAct = PPU0015LinCombDeEltAct<CtaTileShapeMNK, EpilogueTile, Stages, StrideAux, SmemLayoutAtom, CopyOpS2R, ActivationFn, ElementOutput, ElementCompute, ElementAux, ElementSource, ElementScalar, AlignmentAux, RoundStyle>;",
        "template<class CtaTileShapeMNK, class EpilogueTile, int Stages, class StrideAux, class SmemLayoutAtom, class CopyOpS2R, template<class> class ActivationFn, class ElementOutput, class ElementCompute, class ElementAux = ElementOutput, class ElementBias = ElementOutput, class ElementSource = ElementOutput, class ElementScalar = ElementCompute, int AlignmentAux = 128 / sizeof_bits_v<ElementAux>, int AlignmentBias = 128 / sizeof_bits_v<ElementBias>, FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest>\nusing Sm90LinCombDeEltActDePerRowBias = PPU0015LinCombDeEltActDePerRowBias<CtaTileShapeMNK, EpilogueTile, Stages, StrideAux, SmemLayoutAtom, CopyOpS2R, ActivationFn, ElementOutput, ElementCompute, ElementAux, ElementBias, ElementSource, ElementScalar, AlignmentAux, AlignmentBias, RoundStyle>;",
    ]),

        # cutlass::epilogue::fusion — ppu_visitor_load (6+2)
    ("cutlass/epilogue/fusion/ppu_visitor_load_tma_warpspecialized.hpp", "cutlass::epilogue::fusion", [
        "template<int Stages, class EpilogueTile, class Element, class StrideMNL, class SmemLayoutAtom, class CopyOpS2R, int Alignment = 128 / sizeof_bits_v<Element>, bool EnableNullptr = true>\nusing Sm90AuxLoad = PPU0015AuxLoad<Stages, EpilogueTile, Element, StrideMNL, SmemLayoutAtom, CopyOpS2R, Alignment, EnableNullptr>;",
        "template<int Stages, class CtaTileShapeMNK, class Element, class StrideMNL = Stride<_0,_1,_0>, int Alignment = 128 / sizeof_bits_v<Element>, bool EnableNullptr = true>\nusing Sm90RowBroadcast = PPU0015RowBroadcast<Stages, CtaTileShapeMNK, Element, StrideMNL, Alignment, EnableNullptr>;",
        "template<int Stages, class CtaTileShapeMNK, class Element, class StrideMNL = Stride<_1,_0,_0>, int Alignment = 128 / sizeof_bits_v<Element>, bool EnableNullptr = true>\nusing Sm90ColBroadcast = PPU0015ColBroadcast<Stages, CtaTileShapeMNK, Element, StrideMNL, Alignment, EnableNullptr>;",
        "using Sm90SplitTreeFetch = PPU0015AccFetch;",
        "using Sm90AccFetch = PPU0015AccFetch;",
        "template <class Element>\nusing Sm90SrcFetch = PPU0015SrcFetch<Element>;",
        "template<int Stages, class EpilogueTile, class Element, class StrideMNL, class SmemLayoutAtom, class CopyOpS2R, int Alignment = 128 / sizeof_bits_v<Element>, bool EnableNullptr = true>\nusing Sm90MatrixBroadcast = PPU0015AuxLoad<Stages, EpilogueTile, Element, StrideMNL, SmemLayoutAtom, CopyOpS2R, EnableNullptr>;",
        "template<class Element, class StrideMNL = Stride<_0,_0,_0>, int BroadcastCount = 1, template<class> class ReductionFn = multiplies>\nusing Sm90ScalarBroadcast = PPU0015ScalarBroadcast<Element, StrideMNL, BroadcastCount, ReductionFn>;",
    ]),

    # cutlass::epilogue::fusion — ppu_visitor_store (5)
    ("cutlass/epilogue/fusion/ppu_visitor_store_tma_warpspecialized.hpp", "cutlass::epilogue::fusion", [
        "template<int Stages, class EpilogueTile, class Element, FloatRoundStyle RoundStyle, class StrideMNL, class SmemLayoutAtom, class CopyOpR2S, int Alignment = 128 / sizeof_bits_v<Element>, bool EnableNullptr = true>\nusing Sm90AuxStore = PPU0015AuxStore<Stages, EpilogueTile, Element, RoundStyle, StrideMNL, SmemLayoutAtom, CopyOpR2S, Alignment, EnableNullptr>;",
        "template<int Stages, class EpilogueTile, class Element, class StrideMNL, class CopyOpR2S, class SmemLayoutAtom, int Alignment = 128 / sizeof_bits_v<Element>, bool EnableNullptr = true>\nusing Sm90MatrixReduction = PPU0015MatrixReduction<Stages, EpilogueTile, Element, StrideMNL, CopyOpR2S, SmemLayoutAtom, Alignment, EnableNullptr>;",
        "template<template<class> class RegReduceFn, template<class> class GmemReduceFn, class ElementOutput, class ElementCompute, FloatRoundStyle RoundStyle, class StrideMNL = Stride<_0,_0,_0>, bool EnableNullptr = true>\nusing Sm90ScalarReduction = PPU0015ScalarReduction<RegReduceFn, GmemReduceFn, ElementOutput, ElementCompute, RoundStyle, StrideMNL, EnableNullptr>;",
        "template<template<class> class RegReduceFn, template<class> class GmemReduceFn, int Stages, class CtaTileShapeMNK, class ElementOutput, class ElementCompute, FloatRoundStyle RoundStyle, class StrideMNL = Stride<_0,_1,_0>, int Alignment = 128 / sizeof_bits_v<ElementOutput>, bool EnableNullptr = true>\nusing Sm90RowReduction = PPU0015RowReduction<RegReduceFn, GmemReduceFn, Stages, CtaTileShapeMNK, ElementOutput, ElementCompute, RoundStyle, StrideMNL, Alignment, EnableNullptr>;",
        "template<template<class> class RegReduceFn, template<class> class GmemReduceFn, int Stages, class CtaTileShapeMNK, class ElementOutput, class ElementCompute, FloatRoundStyle RoundStyle, class StrideMNL = Stride<_1,_0,_0>, int Alignment = 128 / sizeof_bits_v<ElementOutput>, bool EnableNullptr = true, bool FinalReduction = true>\nusing Sm90ColReduction = PPU0015ColReduction<RegReduceFn, GmemReduceFn, Stages, CtaTileShapeMNK, ElementOutput, ElementCompute, RoundStyle, StrideMNL, Alignment, EnableNullptr, FinalReduction>;",
    ]),

    # # cutlass::epilogue::collective::detail — TMA adapter (1)
    # ("cutlass/epilogue/collective/detail.hpp", "cutlass::epilogue::collective::detail", [
    #     "template<class EpilogueOp>\nusing Sm90TmaWarpSpecializedAdapter = PPU0015TmaWarpSpecializedAdapter<EpilogueOp>;",
    # ]),

    # cutlass::epilogue::fusion — visitor base/impl/tree/split/topo (5)
    ("cutlass/epilogue/fusion/ppu_visitor_tma_warpspecialized.hpp", "cutlass::epilogue::fusion", [
        "template <class... Ops>\nusing Sm90VisitorImplBase = PPU0015VisitorImplBase<Ops...>;",
        "template <class... Ops>\nusing Sm90VisitorImpl = PPU0015VisitorImpl<Ops...>;",
        "template <class InputTree, class OutputTree, class... AuxOutTrees>\nusing Sm90SplitTreeVisitor = PPU0015SplitTreeVisitor<InputTree, OutputTree, AuxOutTrees...>;",
        "template <class ElementCompute, class EdgeTuple, class... Ops>\nusing Sm90TopologicalVisitor = PPU0015TopologicalVisitor<ElementCompute, EdgeTuple, Ops...>;",
    ]),

    # cutlass::epilogue::fusion — compute + ReLU aux store (2)
    ("cutlass/epilogue/fusion/ppu_visitor_compute_tma_warpspecialized.hpp", "cutlass::epilogue::fusion", [
        "template<template <class> class ComputeFn, class ElementOutput, class ElementCompute, FloatRoundStyle RoundStyle, class = void>\nusing Sm90Compute = PPU0015Compute<ComputeFn, ElementOutput, ElementCompute, RoundStyle>;",
        "template<class StrideMNL>\nusing Sm90ReLUAuxStore = PPU0015ReLUAuxStore<StrideMNL>;",
    ]),

    # cutlass::epilogue::fusion — ExtraFetch (PPU extension) (1)
    ("ppu/cutlass/epilogue/fusion/ppu_visitor_load_tma_warpspecialized.hpp", "cutlass::epilogue::fusion", [
        "template <typename Element, int InputIdx>\nusing Sm90ExtraFetch = PPU0015ExtraFetch<Element, InputIdx>;",
    ]),

    # ========================================================================
    # cutlass::gemm::thread::detail — EnableMma_Crow (SM60, 1)
    # ========================================================================
    ("cutlass/gemm/thread/mma_ppu0010.h", "cutlass::gemm::thread::detail", [
        "template <typename LayoutA, typename LayoutB>\nusing EnableMma_Crow_SM60 = EnableMma_Crow_PPU0010<LayoutA, LayoutB>;",
    ]),

    # ========================================================================
    # cutlass — HostAdapter type alias (CudaHostAdapter -> HggcHostAdapter)
    # ========================================================================
    ("cutlass/hggc_host_adapter.hpp", "cutlass", [
        "using CudaHostAdapter = HggcHostAdapter;",
    ]),
]

MARKER = "// CU -> PPU backward compatibility aliases"


def apply(path):
    """Append backward compatibility aliases to source files under the given path."""
    total_aliases = 0
    total_files = 0
    for filepath, namespace, lines in ALIASES:
        full_path = os.path.join(path, filepath)
        if not os.path.exists(full_path):
            print(f"WARNING: {filepath} not found")
            continue
        with open(full_path, 'r') as f:
            content = f.read()
        if MARKER in content:
            print(f"SKIP: {filepath} (already has aliases)")
            continue
        # Build the block
        parts = namespace.split("::")
        ns_open = "namespace " + " { namespace ".join(parts) + " {"
        ns_close = "} " * len(parts) + f"// namespace {namespace}"
        block = f"\n{MARKER}\n{ns_open}\n"
        for line in lines:
            block += f"{line}\n"
        block += f"{ns_close}\n"
        with open(full_path, 'a') as f:
            f.write(block)
        count = len(lines)
        total_aliases += count
        total_files += 1
        print(f"OK: {filepath} ({count} aliases)")
    print(f"\nTotal: {total_aliases} aliases in {total_files} files")


# ============================================================================
# Symbol replacement: PPU -> CU
# ============================================================================
# Certain PPU symbol names must be reverted to their original CU names.
# For example, certain PPU macro names that must remain as their original CU names
# so that the corresponding code paths compile correctly with the original CU semantics.

# Each entry: (hgcc_symbol, cuda_symbol) — replace HGCC back to CUDA.
# Order matters: longer strings must come first to avoid partial matches.
SYMBOL_REPLACES = [
    # ========================================================================
    # SM80: CP_ASYNC macro must remain as SM80 name to avoid NaN at runtime
    # ========================================================================
    ("CUTE_ARCH_CP_ASYNC_PPU_ENABLED", "CUTE_ARCH_CP_ASYNC_SM80_ENABLED"),
    ("CUTE_ARCH_LDSM_PPU_ACTIVATED", "CUTE_ARCH_LDSM_SM75_ACTIVATED"),
    ("CUTE_ARCH_LDSM_PPU_ENABLED", "CUTE_ARCH_LDSM_SM75_ENABLED"),

    # ========================================================================
    # Compiler macros (reverse of COMPILER_MACRO_MAP, excluding __HGGC_ARCH__)
    # Sorted by length descending to avoid partial matches.
    # ========================================================================
    ("__HGGCCC_DIAG_PRAGMA_SUPPORT__", "__NVCC_DIAG_PRAGMA_SUPPORT__"),
    ("__HGGC_NO_HALF2_OPERATORS__", "__CUDA_NO_HALF2_OPERATORS__"),
    ("__HGGC_NO_HALF_OPERATORS__", "__CUDA_NO_HALF_OPERATORS__"),
    ("__HGGCCC_VER_MAJOR__", "__CUDACC_VER_MAJOR__"),
    ("__HGGCCC_VER_MINOR__", "__CUDACC_VER_MINOR__"),
    ("__HGGCCC_VERSION__", "__CUDACC_VERSION__"),
    ("__HGGCCC_RTC__", "__CUDACC_RTC__"),
    ("__HGGCCC__", "__CUDACC__"),
    ("__HGGC__", "__CUDA__"),

    # ========================================================================
    # API identifiers (reverse of IDENT_MAP)
    # Sorted by length descending to avoid partial matches.
    # ========================================================================
    # ---- occupancy ----
    ("hggcOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
     "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags"),
    ("hggcOccupancyMaxActiveBlocksPerMultiprocessor",
     "cudaOccupancyMaxActiveBlocksPerMultiprocessor"),
    ("hggcOccupancyMaxPotentialBlockSize",
     "cudaOccupancyMaxPotentialBlockSize"),
    ("hggcOccupancyDisableCachingOverride",
     "cudaOccupancyDisableCachingOverride"),

    # ---- launch attributes / cluster ----
    ("hggcLaunchAttributeProgrammaticStreamSerialization",
     "cudaLaunchAttributeProgrammaticStreamSerialization"),
    ("hggcLaunchAttributeValue", "cudaLaunchAttributeValue"),
    ("hggcLaunchAttributeID", "cudaLaunchAttributeID"),
    ("hggcLaunchAttribute", "cudaLaunchAttribute"),
    ("hggcLaunchConfig_t", "cudaLaunchConfig_t"),
    ("hggcLaunchKernelExC", "cudaLaunchKernelExC"),
    ("hggcLaunchKernelEx", "cudaLaunchKernelEx"),
    ("hggcLaunchKernel", "cudaLaunchKernel"),

    # ---- device management / func attributes ----
    ("hggcFuncAttributeMaxDynamicSharedMemorySize",
     "cudaFuncAttributeMaxDynamicSharedMemorySize"),
    ("hggcDeviceSetCacheConfig", "cudaDeviceSetCacheConfig"),
    ("hggcDeviceGetAttribute", "cudaDeviceGetAttribute"),
    ("hggcGetDeviceProperties", "cudaGetDeviceProperties"),
    ("hggcDevAttrMultiProcessorCount", "cudaDevAttrMultiProcessorCount"),
    ("hggcFuncGetAttributes", "cudaFuncGetAttributes"),
    ("hggcFuncSetAttribute", "cudaFuncSetAttribute"),
    ("hggcFuncAttributes", "cudaFuncAttributes"),
    ("hggcFuncAttribute", "cudaFuncAttribute"),
    ("hggcDeviceSynchronize", "cudaDeviceSynchronize"),
    ("hggcGetDeviceCount", "cudaGetDeviceCount"),
    ("hggcGetDeviceProp", "cudaGetDeviceProp"),
    ("hggcDeviceProp", "cudaDeviceProp"),
    ("hggcDeviceAttr", "cudaDeviceAttr"),
    ("hggcDeviceReset", "cudaDeviceReset"),
    ("hggcDeviceId", "cudaDeviceId"),
    ("hggcGetDevice", "cudaGetDevice"),
    ("hggcSetDevice", "cudaSetDevice"),
    ("hggcGetNumDevices", "cudaGetNumDevices"),

    # ---- driver entry-point lookup ----
    ("hggcGetDriverEntryPointByVersion", "cudaGetDriverEntryPointByVersion"),
    ("hggcGetDriverEntryPoint", "cudaGetDriverEntryPoint"),
    ("hggcDriverEntryPointQueryResult", "cudaDriverEntryPointQueryResult"),
    ("hggcDriverEntryPointSuccess", "cudaDriverEntryPointSuccess"),
    ("hggcEnableDefault", "cudaEnableDefault"),

    # ---- memory / stream / event ----
    ("hggcMemcpyDeviceToDevice", "cudaMemcpyDeviceToDevice"),
    ("hggcMemcpyDeviceToHost", "cudaMemcpyDeviceToHost"),
    ("hggcMemcpyHostToDevice", "cudaMemcpyHostToDevice"),
    ("hggcMemcpyHostToHost", "cudaMemcpyHostToHost"),
    ("hggcMemcpyFromSymbol", "cudaMemcpyFromSymbol"),
    ("hggcMemcpyToSymbol", "cudaMemcpyToSymbol"),
    ("hggcMemcpyDefault", "cudaMemcpyDefault"),
    ("hggcMemPrefetchAsync", "cudaMemPrefetchAsync"),
    ("hggcStreamCaptureModeGlobal", "cudaStreamCaptureModeGlobal"),
    ("hggcStreamNonBlocking", "cudaStreamNonBlocking"),
    ("hggcStreamPerThread", "cudaStreamPerThread"),
    ("hggcStreamCreateWithFlags", "cudaStreamCreateWithFlags"),
    ("hggcStreamBeginCapture", "cudaStreamBeginCapture"),
    ("hggcStreamEndCapture", "cudaStreamEndCapture"),
    ("hggcStreamSynchronize", "cudaStreamSynchronize"),
    ("hggcStreamWaitEvent", "cudaStreamWaitEvent"),
    ("hggcStreamLegacy", "cudaStreamLegacy"),
    ("hggcStreamDefault", "cudaStreamDefault"),
    ("hggcStreamDestroy", "cudaStreamDestroy"),
    ("hggcStreamCreate", "cudaStreamCreate"),
    ("hggcMemcpyAsync", "cudaMemcpyAsync"),
    ("hggcMallocManaged", "cudaMallocManaged"),
    ("hggcMallocAsync", "cudaMallocAsync"),
    ("hggcMallocHost", "cudaMallocHost"),
    ("hggcMemsetAsync", "cudaMemsetAsync"),
    ("hggcFreeAsync", "cudaFreeAsync"),
    ("hggcFreeHost", "cudaFreeHost"),
    ("hggcHostAllocPortable", "cudaHostAllocPortable"),
    ("hggcHostAllocMapped", "cudaHostAllocMapped"),
    ("hggcHostAllocDefault", "cudaHostAllocDefault"),
    ("hggcHostRegisterDefault", "cudaHostRegisterDefault"),
    ("hggcHostUnregister", "cudaHostUnregister"),
    ("hggcHostRegister", "cudaHostRegister"),
    ("hggcHostAlloc", "cudaHostAlloc"),
    ("hggcMemGetInfo", "cudaMemGetInfo"),
    ("hggcMemcpyKind", "cudaMemcpyKind"),
    ("hggcMemAdvise", "cudaMemAdvise"),
    ("hggcMemcpy", "cudaMemcpy"),
    ("hggcMemset", "cudaMemset"),
    ("hggcMalloc", "cudaMalloc"),
    ("hggcFree", "cudaFree"),

    # ---- event ----
    ("hggcEventCreateWithFlags", "cudaEventCreateWithFlags"),
    ("hggcEventRecordWithFlags", "cudaEventRecordWithFlags"),
    ("hggcEventRecordDefault", "cudaEventRecordDefault"),
    ("hggcEventRecordExternal", "cudaEventRecordExternal"),
    ("hggcEventDisableTiming", "cudaEventDisableTiming"),
    ("hggcEventBlockingSync", "cudaEventBlockingSync"),
    ("hggcEventSynchronize", "cudaEventSynchronize"),
    ("hggcEventElapsedTime", "cudaEventElapsedTime"),
    ("hggcEventElapsed", "cudaEventElapsed"),
    ("hggcEventDestroy", "cudaEventDestroy"),
    ("hggcEventRecord", "cudaEventRecord"),
    ("hggcEventCreate", "cudaEventCreate"),
    ("hggcEventQuery", "cudaEventQuery"),
    ("hggcEvent_t", "cudaEvent_t"),

    # ---- error / status ----
    ("hggcErrorMemoryAllocation", "cudaErrorMemoryAllocation"),
    ("hggcErrorInvalidValue", "cudaErrorInvalidValue"),
    ("hggcGetErrorString", "cudaGetErrorString"),
    ("hggcGetErrorName", "cudaGetErrorName"),
    ("hggcPeekAtLastError", "cudaPeekAtLastError"),
    ("hggcGetLastError", "cudaGetLastError"),
    ("hggcErrorUnknown", "cudaErrorUnknown"),
    ("hggcError_t", "cudaError_t"),
    ("hggcError", "cudaError"),
    ("hggcSuccess", "cudaSuccess"),

    # ---- stream / graph ----
    ("hggcStream_t", "cudaStream_t"),
    ("hggcGraphExecDestroy", "cudaGraphExecDestroy"),
    ("hggcGraphInstantiate", "cudaGraphInstantiate"),
    ("hggcGraphDestroy", "cudaGraphDestroy"),
    ("hggcGraphLaunch", "cudaGraphLaunch"),
    ("hggcGraphExec_t", "cudaGraphExec_t"),
    ("hggcGraph_t", "cudaGraph_t"),

    # ---- misc memory / type / device ----
    ("hggcCreateChannelDesc", "cudaCreateChannelDesc"),
    ("hggcChannelFormatDesc", "cudaChannelFormatDesc"),
    ("hggcDestroyTextureObject", "cudaDestroyTextureObject"),
    ("hggcCreateTextureObject", "cudaCreateTextureObject"),
    ("hggcResourceTypeArray", "cudaResourceTypeArray"),
    ("hggcPointerGetAttributes", "cudaPointerGetAttributes"),
    ("hggcPointerAttributes", "cudaPointerAttributes"),
    ("hggcTextureObject_t", "cudaTextureObject_t"),
    ("hggcDataType_t", "cudaDataType_t"),
    ("hggcResourceDesc", "cudaResourceDesc"),
    ("hggcTextureDesc", "cudaTextureDesc"),
    ("hggcArrayDefault", "cudaArrayDefault"),
    ("hggcDataType", "cudaDataType"),
    ("hggcArray_t", "cudaArray_t"),

    # ---- driver API (cu* prefix) → reverse ----
    ("hgMemsetD32Async", "cuMemsetD32Async"),
    ("hgMemsetD16Async", "cuMemsetD16Async"),
    ("hgMemsetD8Async", "cuMemsetD8Async"),
    ("hgGetErrorString", "cuGetErrorString"),
    ("hgGetErrorName", "cuGetErrorName"),
    ("HGGC_ERROR_UNKNOWN", "CUDA_ERROR_UNKNOWN"),
    ("HGGC_SUCCESS", "CUDA_SUCCESS"),

    # ---- tensormap (Hopper TMA) ----
    # Enum values (UPPER_CASE with underscores) — must come before type names
    ("HG_TENSOR_MAP_", "CU_TENSOR_MAP_"),
    # Type names (camelCase)
    ("HGtensorMapFloatOOBfill", "CUtensorMapFloatOOBfill"),
    ("HGtensorMapL2promotion", "CUtensorMapL2promotion"),
    ("HGtensorMapInterleave", "CUtensorMapInterleave"),
    ("HGtensorMapDataType", "CUtensorMapDataType"),
    ("hgTensorMapEncodeTiled", "cuTensorMapEncodeTiled"),
    ("HGtensorMapSwizzle", "CUtensorMapSwizzle"),
    ("HGtensorMap", "CUtensorMap"),

    # ---- result / driver ----
    ("HGdeviceptr", "CUdeviceptr"),
    ("HGfunction", "CUfunction"),
    ("HGcontext", "CUcontext"),
    ("HGmodule", "CUmodule"),
    ("HGstream", "CUstream"),
    ("HGdevice", "CUdevice"),
    ("HGresult", "CUresult"),

    # ---- data type enums ----
    ("HGGC_C_64F", "CUDA_C_64F"),
    ("HGGC_C_32F", "CUDA_C_32F"),
    ("HGGC_C_16F", "CUDA_C_16F"),
    ("HGGC_R_64F", "CUDA_R_64F"),
    ("HGGC_R_32I", "CUDA_R_32I"),
    ("HGGC_R_32F", "CUDA_R_32F"),
    ("HGGC_R_16F", "CUDA_R_16F"),
    ("HGGC_R_8I", "CUDA_R_8I"),

    # ---- cublas typed enums (reverse: ACBLAS → CUBLAS) ----
    ("ACBLAS_GEMM_DEFAULT_TENSOR_OP", "CUBLAS_GEMM_DEFAULT_TENSOR_OP"),
    ("ACBLAS_STATUS_NOT_INITIALIZED", "CUBLAS_STATUS_NOT_INITIALIZED"),
    ("ACBLAS_STATUS_EXECUTION_FAILED", "CUBLAS_STATUS_EXECUTION_FAILED"),
    ("ACBLAS_STATUS_ALLOC_FAILED", "CUBLAS_STATUS_ALLOC_FAILED"),
    ("ACBLAS_STATUS_INVALID_VALUE", "CUBLAS_STATUS_INVALID_VALUE"),
    ("ACBLAS_STATUS_ARCH_MISMATCH", "CUBLAS_STATUS_ARCH_MISMATCH"),
    ("ACBLAS_STATUS_MAPPING_ERROR", "CUBLAS_STATUS_MAPPING_ERROR"),
    ("ACBLAS_STATUS_INTERNAL_ERROR", "CUBLAS_STATUS_INTERNAL_ERROR"),
    ("ACBLAS_STATUS_NOT_SUPPORTED", "CUBLAS_STATUS_NOT_SUPPORTED"),
    ("ACBLAS_STATUS_LICENSE_ERROR", "CUBLAS_STATUS_LICENSE_ERROR"),
    ("ACBLAS_STATUS_SUCCESS", "CUBLAS_STATUS_SUCCESS"),
    ("ACBLAS_ERROR", "CUBLAS_ERROR"),

    # ---- cuComplex API (reverse: ac → cu) ----
    ("acComplex.h", "cuComplex.h"),
    ("make_acDoubleComplex", "make_cuDoubleComplex"),
    ("make_acFloatComplex", "make_cuFloatComplex"),
    ("acDoubleComplex", "cuDoubleComplex"),
    ("acFloatComplex", "cuFloatComplex"),
    ("acCimagf", "cuCimagf"),
    ("acCrealf", "cuCrealf"),
    ("acCimag", "cuCimag"),
    ("acCreal", "cuCreal"),

    # ---- cublas / cudnn type & function names (reverse) ----
    ("acblasOperation_t", "cublasOperation_t"),
    ("acblasStatus_t", "cublasStatus_t"),
    ("acblasHandle_t", "cublasHandle_t"),
    ("acblasGemmEx", "cublasGemmEx"),
    ("acblasSgemm", "cublasSgemm"),
    ("acblasDgemm", "cublasDgemm"),
    ("acblasHgemm", "cublasHgemm"),
    ("acblasCgemm", "cublasCgemm"),
    ("acblasZgemm", "cublasZgemm"),

    # ---- curand type & function names (reverse) ----
    ("acrandStateScrambledSobol64_t", "curandStateScrambledSobol64_t"),
    ("acrandStateScrambledSobol32_t", "curandStateScrambledSobol32_t"),
    ("acrandDirectionVectors64_t", "curandDirectionVectors64_t"),
    ("acrandDirectionVectors32_t", "curandDirectionVectors32_t"),
    ("acrandStateMRG32k3a_t", "curandStateMRG32k3a_t"),
    ("acrandStateSobol32_t", "curandStateSobol32_t"),
    ("acrandStateSobol64_t", "curandStateSobol64_t"),
    ("acrandStateXORWOW_t", "curandStateXORWOW_t"),
    ("acrandStateXORWOW", "curandStateXORWOW"),
    ("acrandStateTest_t", "curandStateTest_t"),
    ("acrand_uniform_double", "curand_uniform_double"),
    ("acrand_normal_double", "curand_normal_double"),
    ("acrand_log_normal_double", "curand_log_normal_double"),
    ("acrand_log_normal", "curand_log_normal"),
    ("acrand_uniform", "curand_uniform"),
    ("acrand_normal", "curand_normal"),
    ("acrand_poisson", "curand_poisson"),
    ("acrandState_t", "curandState_t"),
    ("acrandState", "curandState"),
    ("acrand_init", "curand_init"),

    # ---- logging helpers (reverse) ----
    ("PPU_PERROR_EXIT", "CUDA_PERROR_EXIT"),
    ("PPU_PERROR_DEBUG", "CUDA_PERROR_DEBUG"),
    ("PPU_LOG_DEBUG", "CUDA_LOG_DEBUG"),
    ("PPU_PERROR", "CUDA_PERROR"),
    ("PPU_LOG", "CUDA_LOG"),

    # ---- cutlass / cute internal gating identifiers (reverse) ----
    ("CUTE_STL_NAMESPACE_IS_HGGC_STD", "CUTE_STL_NAMESPACE_IS_CUDA_STD"),
    ("CUTLASS_ENABLE_HOST_ADAPTER", "CUTLASS_ENABLE_CUDA_HOST_ADAPTER"),
    ("CUTE_PPU_SUPPORTS_CVTA_GENERIC_TO_SHARED",
     "CUTE_NVCC_SUPPORTS_CVTA_GENERIC_TO_SHARED"),
    ("CUTE_PPU_SUPPORTS_GET_SMEM_POINTER",
     "CUTE_NVCC_SUPPORTS_NVVM_GET_SMEM_POINTER"),
    ("CUTE_CLANG_SUPPORTS_GET_SMEM_POINTER",
     "CUTE_CLANG_SUPPORTS_NVVM_GET_SMEM_POINTER"),
    ("CUTE_GET_SMEM_POINTER_SUPPORTED",
     "CUTE_NVVM_GET_SMEM_POINTER_SUPPORTED"),
    ("CUTE_GET_SMEM_POINTER_ACTIVATED",
     "CUTE_NVVM_GET_SMEM_POINTER_ACTIVATED"),

    # ---- local-variable / helper names (reverse) ----
    ("device_perror_impl", "cuda_perror_impl"),
    ("device_exception", "cuda_exception"),

    # ---- self-defined macros (reverse) ----
    ("CUTLASS_PPU_CHECK", "CUDA_CHECK"),

    # ---- NV vendor-prefixed numeric typedefs (reverse) ----
    ("__ppu_bfloat162_raw", "__nv_bfloat162_raw"),
    ("__ppu_bfloat16_raw", "__nv_bfloat16_raw"),
    ("__hg_fp8_storage_t", "__nv_fp8_storage_t"),
    ("to_ppu_bfloat16", "to_nv_bfloat16"),
    ("__ppu_bfloat162", "__nv_bfloat162"),
    ("__hg_fp8_e4m3", "__nv_fp8_e4m3"),
    ("__hg_fp8_e5m2", "__nv_fp8_e5m2"),
    ("ppu_bfloat162", "nv_bfloat162"),
    ("ppu_bfloat16", "nv_bfloat16"),
    ("__ppu_bfloat16", "__nv_bfloat16"),

    # ---- std-clamp wrappers (reverse) ----
    ("__HGGC_STD_XYZ", "__NV_STD_XYZ"),
    ("__HGGC_STD_MIN", "__NV_STD_MIN"),
    ("__HGGC_STD_MAX", "__NV_STD_MAX"),
    ("__HGGC_STD_", "__NV_STD_"),

    # ========================================================================
    # Header paths (reverse of HEADER_INCLUDE_MAP)
    # Sorted by length descending.
    # ========================================================================
    ("hggc_awbarrier_primitives.h", "cuda_awbarrier_primitives.h"),
    ("hggc_pipeline_primitives.h", "cuda_pipeline_primitives.h"),
    ("cutlass/util/acblas_wrappers.hpp", "cutlass/util/cublas_wrappers.hpp"),
    ("hggc_awbarrier.h", "cuda_awbarrier.h"),
    ("hggcrt_driver_types.h", "driver_types.h"),
    ("hggc_pipeline.h", "cuda_pipeline.h"),
    ("hggc_vector_types.h", "vector_types.h"),
    ("hggcTypedefs.h", "cudaTypedefs.h"),
    ("hggc_runtime_api.h", "cuda_runtime_api.h"),
    ("hggc_runtime.h", "cuda_runtime.h"),
    ("hggc_fp16.hpp", "cuda_fp16.hpp"),
    ("hggc_bf16.hpp", "cuda_bf16.hpp"),
    ("hggc_fp16.h", "cuda_fp16.h"),
    ("hggc_bf16.h", "cuda_bf16.h"),
    ("hggc_fp8.h", "cuda_fp8.h"),
    ("hggc.h", "cuda.h"),

    # ========================================================================
    # Header prefixes (reverse of HEADER_PREFIX_MAP)
    # Sorted by length descending.
    # ========================================================================
    ("hggc/memory_resource", "cuda/memory_resource"),
    ("hggc/annotated_ptr", "cuda/annotated_ptr"),
    ("hggc/stream_ref", "cuda/stream_ref"),
    ("hggc/functional", "cuda/functional"),
    ("hggc/semaphore", "cuda/semaphore"),
    ("hggc/pipeline", "cuda/pipeline"),
    ("hggc/barrier", "cuda/barrier"),
    ("hggc/atomic", "cuda/atomic"),
    ("hggc/std/", "cuda/std/"),
    ("hggc/latch", "cuda/latch"),
]

REPLACE_MARKER = "// PPU to CU symbol replacement"

# File extensions to process
SOURCE_EXTENSIONS = ('.hpp', '.h', '.inl', '.cu')


def replace_symbols(path):
    """Replace PPU symbols with CU equivalents in source files under the given path.

    Walks the directory tree at `path` and replaces all occurrences of
    old_symbol with new_symbol for each entry in SYMBOL_REPLACES.

    Idempotent: if old_symbol does not exist in a file, it is left untouched.
    Running multiple times produces the same result.
    """
    total_files = 0

    for root, dirs, files in os.walk(path):
        for filename in files:
            if not filename.endswith(SOURCE_EXTENSIONS):
                continue
            filepath = os.path.join(root, filename)

            with open(filepath, 'r') as f:
                content = f.read()

            new_content = content
            file_changed = False

            for old_symbol, new_symbol in SYMBOL_REPLACES:
                if old_symbol in new_content:
                    new_content = new_content.replace(old_symbol, new_symbol)
                    file_changed = True

            if file_changed:
                with open(filepath, 'w') as f:
                    f.write(new_content)
                total_files += 1
                rel_path = os.path.relpath(filepath, path)
                print(f"REPLACE: {rel_path}")

    if total_files == 0:
        print("No symbols found to replace (already replaced or no matches)")
    else:
        print(f"\nTotal: replaced symbols in {total_files} files")


# ============================================================================
# File name compatibility shims
# ============================================================================
# Certain header files have been renamed (cuda_* -> hggc_*).
# This module creates thin "shim" files at the old file names that simply
# #include the new file, providing backward compatibility for external code
# that still references the old names.

# Each entry: (old_relative_path, new_relative_path)
FILE_MAPPINGS = [
    ("cute/container/cuda_types.hpp", "cute/container/hggc_types.hpp"),
    ("cutlass/cuda_host_adapter.hpp", "cutlass/hggc_host_adapter.hpp"),
    ("cutlass/floating_point_nvrtc.h", "cutlass/floating_point_hgrtc.h"),

    # --- Merge group M1: mma_sm70/75/80 → mma_ppu ---
    ("cutlass/arch/mma_sm70.h", "cutlass/arch/mma_ppu0010.h"),
    ("cutlass/arch/mma_sm75.h", "cutlass/arch/mma_ppu0010.h"),
    ("cutlass/arch/mma_sm80.h", "cutlass/arch/mma_ppu0010.h"),
    # --- Merge group M2: memory_sm75/80 → memory_ppu ---
    ("cutlass/arch/memory_sm75.h", "cutlass/arch/memory_ppu.h"),
    ("cutlass/arch/memory_sm80.h", "cutlass/arch/memory_ppu.h"),

    # --- Single file renames ---
    ("cutlass/epilogue/collective/sm90_epilogue_tma_warpspecialized.hpp", "cutlass/epilogue/collective/ppu_epilogue_tma_warpspecialized.hpp"),
    ("cutlass/epilogue/collective/sm90_epilogue_tma_warpspecialized_bias_elementwise.hpp", "cutlass/epilogue/collective/ppu_epilogue_tma_warpspecialized_bias_elementwise.hpp"),
    ("cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp", "cutlass/epilogue/fusion/ppu_callbacks_tma_warpspecialized.hpp"),
    ("cutlass/epilogue/fusion/sm90_visitor_compute_tma_warpspecialized.hpp", "cutlass/epilogue/fusion/ppu_visitor_compute_tma_warpspecialized.hpp"),
    ("cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp", "cutlass/epilogue/fusion/ppu_visitor_load_tma_warpspecialized.hpp"),
    ("cutlass/epilogue/fusion/sm90_visitor_store_tma_warpspecialized.hpp", "cutlass/epilogue/fusion/ppu_visitor_store_tma_warpspecialized.hpp"),
    ("cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp", "cutlass/epilogue/fusion/ppu_visitor_tma_warpspecialized.hpp"),
    ("cutlass/epilogue/collective/builders/sm90_builder.inl", "cutlass/epilogue/collective/builders/ppu_builder.inl"),
    ("cutlass/gemm/kernel/sm90_tile_scheduler.hpp", "cutlass/gemm/kernel/persistent_tile_scheduler.hpp"),
    ("cutlass/gemm/kernel/sm90_tile_scheduler_group.hpp", "cutlass/gemm/kernel/ppu_tile_scheduler_group.hpp"),
    ("cutlass/gemm/kernel/sm90_tile_scheduler_stream_k.hpp", "cutlass/gemm/kernel/ppu_tile_scheduler_stream_k.hpp"),

    # --- PPU directory file renames ---
    ("ppu/cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp", "ppu/cutlass/epilogue/fusion/ppu_visitor_load_tma_warpspecialized.hpp"),

    # --- Merge group M6: mma_sm50/60/61 → mma_ppu ---
    ("cutlass/arch/mma_sm50.h", "cutlass/arch/mma_ppu0010.h"),
    ("cutlass/arch/mma_sm60.h", "cutlass/arch/mma_ppu0010.h"),
    ("cutlass/arch/mma_sm61.h", "cutlass/arch/mma_ppu0010.h"),

    # cute/ CU file renames - merge groups
    ("cute/arch/mma_sm61.hpp", "cute/arch/mma_ppu.hpp"),
    ("cute/arch/mma_sm70.hpp", "cute/arch/mma_ppu.hpp"),
    ("cute/arch/mma_sm75.hpp", "cute/arch/mma_ppu.hpp"),
    ("cute/arch/mma_sm80.hpp", "cute/arch/mma_ppu.hpp"),
    ("cute/atom/mma_traits_sm61.hpp", "cute/atom/mma_traits_ppu.hpp"),
    ("cute/atom/mma_traits_sm70.hpp", "cute/atom/mma_traits_ppu.hpp"),
    ("cute/atom/mma_traits_sm75.hpp", "cute/atom/mma_traits_ppu.hpp"),
    ("cute/atom/mma_traits_sm80.hpp", "cute/atom/mma_traits_ppu.hpp"),
    ("cute/arch/copy_sm75.hpp", "cute/arch/copy_ppu.hpp"),
    ("cute/arch/copy_sm80.hpp", "cute/arch/copy_ppu.hpp"),
    ("cute/atom/copy_traits_sm75.hpp", "cute/atom/copy_traits_ppu.hpp"),
    ("cute/atom/copy_traits_sm80.hpp", "cute/atom/copy_traits_ppu.hpp"),
]

FILE_MAP_MARKER = "// File name compatibility shim"


def create_file_shims(path):
    """Create thin shim headers at old file names that #include the new file.

    For each (old, new) pair, if the new file exists, write a shim file at the
    old path containing just the marker comment and an #include of the new file.

    Idempotent: if the old file already contains the shim marker, it is skipped.
    """
    total_shims = 0
    for old_rel, new_rel in FILE_MAPPINGS:
        old_path = os.path.join(path, old_rel)
        new_path = os.path.join(path, new_rel)
        if not os.path.exists(new_path):
            print(f"SKIP: {new_rel} (new file not found)")
            continue
        old_dir = os.path.dirname(old_path)
        new_basename = os.path.basename(new_rel)
        shim_content = f'{FILE_MAP_MARKER}\n#include "{new_basename}"\n'
        if os.path.exists(old_path):
            with open(old_path, 'r') as f:
                if FILE_MAP_MARKER in f.read():
                    print(f"SKIP: {old_rel} (shim already exists)")
                    continue
        with open(old_path, 'w') as f:
            f.write(shim_content)
        print(f"SHIM: {old_rel} -> {new_rel}")
        total_shims += 1

    print(f"\nTotal: {total_shims} file shims created")


ARCH_COMPAT_HEADER = "cute/hggc_arch_compat.h"
ARCH_COMPAT_MARKER = "__HGGC_ARCH__"

ARCH_COMPAT_CONTENT = """\
// HGCC Architecture compatibility header
// Auto-generated by cuda_compat_v0.8.0.py - DO NOT COMMIT
// Maps __HGGC_ARCH__ based on __CUDA_ARCH__ for CUDA compilation
#pragma once

#if defined(__CUDA_ARCH__) && !defined(__HGGC_ARCH__)
#if __CUDA_ARCH__ >= 890
#define __HGGC_ARCH__ 150
#else
#define __HGGC_ARCH__ 100
#endif
#endif
"""


def inject_arch_header(path):
    """Create hggc_arch_compat.h and inject #include into files using __HGGC_ARCH__.

    Step 1: Create the compatibility header file (if it does not already exist).
    Step 2: Recursively scan path for .hpp/.h/.inl files that reference
            __HGGC_ARCH__ and inject the compatibility header include after
            #pragma once.

    Idempotent: files that already contain the include are skipped.
    """
    # Step 1: Create the compat header file
    compat_header_path = os.path.join(path, ARCH_COMPAT_HEADER)
    os.makedirs(os.path.dirname(compat_header_path), exist_ok=True)
    with open(compat_header_path, 'w') as f:
        f.write(ARCH_COMPAT_CONTENT)
    print(f"Created {compat_header_path}")

    # Step 2: Inject #include in files that use __HGGC_ARCH__
    INCLUDE_MARKER = "hggc_arch_compat.h"
    total_injected = 0
    for root, dirs, files in os.walk(path):
        for fname in files:
            if not fname.endswith(('.hpp', '.h', '.inl')):
                continue
            fpath = os.path.join(root, fname)
            # Skip the compat header itself
            if os.path.abspath(fpath) == os.path.abspath(compat_header_path):
                continue
            with open(fpath, 'r') as f:
                content = f.read()
            if ARCH_COMPAT_MARKER not in content:
                continue
            if INCLUDE_MARKER in content:
                continue  # Already injected
            # Determine include style
            if '#include "' in content:
                inc_line = '#include "cute/hggc_arch_compat.h"'
            else:
                inc_line = '#include <cute/hggc_arch_compat.h>'
            # Insert after #pragma once
            lines = content.split('\n')
            new_lines = []
            injected = False
            for i, line in enumerate(lines):
                new_lines.append(line)
                if not injected and line.strip() == '#pragma once':
                    new_lines.append(inc_line)
                    injected = True
            if injected:
                with open(fpath, 'w') as f:
                    f.write('\n'.join(new_lines))
                total_injected += 1
                rel_path = os.path.relpath(fpath, path)
                print(f"INJECT: {rel_path}")

    print(f"\nTotal: injected arch compat header in {total_injected} files")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "."
    apply(path)
    replace_symbols(path)
    create_file_shims(path)
    inject_arch_header(path)
