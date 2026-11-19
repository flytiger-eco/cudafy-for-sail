"""ACTLIZE 1.0.0 library compatibility data.
Source: official cudafy pytorch fe5b4338bd37bcc142c7609a2b99af50b4345935.
Library rules only; general remains the sole SDK conversion engine.
"""

VERSION = '1.0.0'

ALIASES = {'cutlass/ppu_host_adapter.hpp': ('cutlass',
                                  '// Host adapter type aliases\n'
                                  'using CudaHostLaunchAttributes = HostLaunchAttributes;\n'
                                  'using CudaHostAdapter = HostAdapter;\n'),
 'cutlass/arch/arch.h': ('cutlass::arch',
                         '// SM architecture tag aliases\n'
                         'using Sm50 = cutlass::arch::PPU0010;\n'
                         'using Sm60 = cutlass::arch::PPU0010;\n'
                         'using Sm61 = cutlass::arch::PPU0010;\n'
                         'using Sm70 = cutlass::arch::PPU0010;\n'
                         'using Sm72 = cutlass::arch::PPU0010;\n'
                         'using Sm75 = cutlass::arch::PPU0010;\n'
                         'using Sm80 = cutlass::arch::PPU0010;\n'
                         'using Sm86 = cutlass::arch::PPU0010;\n'
                         'using Sm89 = cutlass::arch::PPU0015;\n'
                         'using Sm90 = cutlass::arch::PPU0015;\n'
                         '\n'
                         '// SmId function alias\n'
                         'CUTLASS_DEVICE\n'
                         'int SmId() { return cutlass::arch::CuId(); }\n'),
 'cute/arch/copy_ppu.hpp': ('cute',
                            '// SM75 ldmatrix macro aliases\n'
                            '#ifndef CUTE_ARCH_LDSM_SM75_ENABLED\n'
                            '#define CUTE_ARCH_LDSM_SM75_ENABLED CUTE_ARCH_LDSM_PPU_ENABLED\n'
                            '#endif\n'
                            '#ifndef CUTE_ARCH_LDSM_SM75_ACTIVATED\n'
                            '#define CUTE_ARCH_LDSM_SM75_ACTIVATED CUTE_ARCH_LDSM_PPU_ACTIVATED\n'
                            '#endif\n'
                            '// SM80 cp.async macro alias\n'
                            '#ifndef CUTE_ARCH_CP_ASYNC_SM80_ENABLED\n'
                            '#define CUTE_ARCH_CP_ASYNC_SM80_ENABLED CUTE_ARCH_CP_ASYNC_PPU_ENABLED\n'
                            '#endif\n'
                            '\n'
                            '// SM75 ldmatrix\n'
                            'using SM75_U32x1_LDSM_N = PPU_U32x1_LDSM_N;\n'
                            'using SM75_U32x2_LDSM_N = PPU_U32x2_LDSM_N;\n'
                            'using SM75_U32x4_LDSM_N = PPU_U32x4_LDSM_N;\n'
                            'using SM75_U16x2_LDSM_T = PPU_U16x2_LDSM_T;\n'
                            'using SM75_U16x4_LDSM_T = PPU_U16x4_LDSM_T;\n'
                            'using SM75_U16x8_LDSM_T = PPU_U16x8_LDSM_T;\n'
                            '// SM80 cp.async (class templates)\n'
                            'template <class TS, class TD = TS> using SM80_CP_ASYNC_CACHEALWAYS = '
                            'PPU_CP_ASYNC_CACHEALWAYS<TS, TD>;\n'
                            'template <class TS, class TD = TS> using SM80_CP_ASYNC_CACHEGLOBAL = '
                            'PPU_CP_ASYNC_CACHEGLOBAL<TS, TD>;\n'
                            'template <class TS, class TD = TS> using SM80_CP_ASYNC_CACHEALWAYS_ZFILL = '
                            'PPU_CP_ASYNC_CACHEALWAYS_ZFILL<TS, TD>;\n'
                            'template <class TS, class TD = TS> using SM80_CP_ASYNC_CACHEGLOBAL_ZFILL = '
                            'PPU_CP_ASYNC_CACHEGLOBAL_ZFILL<TS, TD>;\n'),
 'cute/arch/mma_ppu.hpp': ('cute', '// SM61 DP4A\nusing SM61_DP4A = PPU_DP4A;\n'),
 'cutlass/epilogue/dispatch_policy.hpp': ('cutlass::epilogue',
                                          'template <int StagesC_, int StagesD_, int FragmentSize_, bool '
                                          'ReuseSmemC_, bool DelayTmaStore_>\n'
                                          'using Sm90TmaWarpSpecialized = PPUTmaWarpSpecialized<StagesC_, '
                                          'StagesD_, FragmentSize_, ReuseSmemC_, DelayTmaStore_>;\n'
                                          'template <int StagesC_, int StagesD_, int FragmentSize_, bool '
                                          'ReuseSmemC_, bool DelayTmaStore_, int NumEpilogueWarpGroups_>\n'
                                          'using Sm90PtrArrayTmaWarpSpecialized = '
                                          'PPUPtrArrayTmaWarpSpecialized<StagesC_, StagesD_, FragmentSize_, '
                                          'ReuseSmemC_, DelayTmaStore_, NumEpilogueWarpGroups_>;\n'
                                          'template <int StagesC_, int StagesD_, int FragmentSize_ = 2>\n'
                                          'using Sm90TmaWarpSpecializedBiasElementwise = '
                                          'PPUTmaWarpSpecializedBiasElementwise<StagesC_, StagesD_, '
                                          'FragmentSize_>;\n'),
 'cutlass/epilogue/collective/detail.hpp': ('cutlass::epilogue::collective::detail',
                                            'template<class Schedule>\n'
                                            'static constexpr bool sm90_is_ptr_array_tma_v = '
                                            'ppu_is_ptr_array_tma_v<Schedule>;\n'),
 'cutlass/epilogue/fusion/ppu_visitor_tma_warpspecialized.hpp': ('cutlass::epilogue::fusion::detail',
                                                                 'template <class... Ops>\n'
                                                                 'using Sm90VisitorImplBase = '
                                                                 'PPUVisitorImplBase<Ops...>;\n'),
 'cutlass/epilogue/threadblock/fusion/visitor_2x.hpp': ('cutlass::epilogue::threadblock',
                                                        'template <class NodeOp, class... ChildOps>\n'
                                                        'using Sm80EVT = PPUEVT2x<NodeOp, ChildOps...>;\n'
                                                        'template <class ElementCompute, class EdgeTuple, '
                                                        'class... Ops>\n'
                                                        'using Sm80TopologicalVisitor = '
                                                        'PPUTopologicalVisitor<ElementCompute, EdgeTuple, '
                                                        'Ops...>;\n'),
 'cutlass/gemm/dispatch_policy.hpp': ('cutlass::gemm',
                                      'using MainloopSm70TwoStageUnpredicated = '
                                      'MainloopPPUTwoStageUnpredicated;\n'
                                      'using MainloopSm70TwoStage = MainloopPPUTwoStage;\n'
                                      'template<int Stages_>\n'
                                      'using MainloopSm80CpAsyncUnpredicated = '
                                      'MainloopPPUCpAsyncUnpredicated<Stages_>;\n'
                                      'template<int Stages_, class ClusterShape_ = Shape<_1,_1,_1>>\n'
                                      'using MainloopSm80CpAsync = MainloopPPUCpAsyncLegacy<Stages_, '
                                      'ClusterShape_>;\n'
                                      'template<int Stages_>\n'
                                      'using MainloopSm80CpAsyncWithScale = '
                                      'MainloopPPUCpAsyncWithScale<Stages_>;\n'
                                      'template<int Stages_, typename Schedule_ = KernelMultistage>\n'
                                      'using MainloopSm80CpAsyncBatchArray = '
                                      'MainloopPPUCpAsyncBatchArray<Stages_, Schedule_>;\n'
                                      'using MainloopSm70TwoStageUnpredicatedLdmatrix = '
                                      'MainloopPPUTwoStageUnpredicatedLdmatrix;\n'
                                      'using MainloopSm70TwoStageLdmatrix = MainloopPPUTwoStageLdmatrix;\n'
                                      'using MainloopSm70TwoStageLdmatrixBatchArray = '
                                      'MainloopPPUTwoStageLdmatrixBatchArray;\n'),
 'cutlass/gemm/kernel/tile_scheduler_params.h': ('cutlass::gemm::kernel::detail',
                                                 'using PersistentTileSchedulerSm90Params = '
                                                 'PersistentTileSchedulerPPUParams;\n'
                                                 'using PersistentTileSchedulerSm90StreamKParams = '
                                                 'PersistentTileSchedulerPPUStreamKParams;\n'
                                                 'template<class ProblemShape>\n'
                                                 'using PersistentTileSchedulerSm90GroupParams = '
                                                 'PersistentTileSchedulerPPUGroupParams<ProblemShape>;\n'),
 'cutlass/gemm/kernel/tile_scheduler.hpp': ('cutlass::gemm::kernel::detail',
                                            'using PersistentTileSchedulerSm90 = '
                                            'PersistentTileSchedulerPPU;\n'
                                            'template <class TileShape, class ClusterShape>\n'
                                            'using PersistentTileSchedulerSm90StreamK = '
                                            'PersistentTileSchedulerPPUStreamK<TileShape, ClusterShape>;\n'
                                            'template <class GroupProblemShape>\n'
                                            'using PersistentTileSchedulerSm90Group = '
                                            'PersistentTileSchedulerPPUGroup<GroupProblemShape>;\n')}

WRAPPERS = [('cutlass/cuda_host_adapter.hpp', ['cutlass/ppu_host_adapter.hpp']),
 ('cutlass/floating_point_nvrtc.h', ['cutlass/floating_point_hgrtc.h']),
 ('cute/arch/copy_sm75.hpp', ['cute/arch/copy_ppu.hpp']),
 ('cute/arch/copy_sm80.hpp', ['cute/arch/copy_ppu.hpp']),
 ('cute/arch/mma_sm61.hpp', ['cute/arch/mma_ppu.hpp']),
 ('cute/atom/copy_traits_sm75.hpp', ['cute/arch/copy_sm75.hpp', 'cute/atom/copy_traits_ppu.hpp']),
 ('cute/atom/copy_traits_sm80.hpp', ['cute/arch/copy_sm80.hpp', 'cute/atom/copy_traits_ppu.hpp']),
 ('cute/atom/mma_traits_sm61.hpp', ['cute/arch/mma_sm61.hpp', 'cute/atom/mma_traits_ppu.hpp']),
 ('cute/container/cuda_types.hpp', ['cute/container/ppu_types.hpp']),
 ('cutlass/arch/memory_sm75.h', ['cutlass/arch/memory_ppu.h']),
 ('cutlass/arch/memory_sm80.h', ['cutlass/arch/memory_ppu.h']),
 ('cutlass/arch/mma_sm60.h', ['cutlass/arch/mma_ppu.h']),
 ('cutlass/arch/mma_sm61.h', ['cutlass/arch/mma_ppu.h']),
 ('cutlass/arch/simd_sm60.h', ['cutlass/arch/simd_ppu.h']),
 ('cutlass/arch/simd_sm61.h', ['cutlass/arch/simd_ppu.h']),
 ('cutlass/epilogue/collective/sm70_epilogue_vectorized.hpp',
  ['cutlass/epilogue/collective/ppu_epilogue_vectorized.hpp']),
 ('cutlass/epilogue/collective/sm70_epilogue_vectorized_array.hpp',
  ['cutlass/epilogue/collective/ppu_epilogue_vectorized_array.hpp']),
 ('cutlass/epilogue/collective/sm70_epilogue_vectorized_evt.hpp',
  ['cutlass/epilogue/collective/ppu_epilogue_vectorized_evt.hpp']),
 ('cutlass/epilogue/collective/sm70_epilogue_vectorized_parallel.hpp',
  ['cutlass/epilogue/collective/ppu_epilogue_vectorized_parallel.hpp']),
 ('cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp',
  ['cutlass/epilogue/fusion/ppu_visitor_tma_warpspecialized.hpp']),
 ('cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp',
  ['cutlass/epilogue/fusion/ppu_visitor_load_tma_warpspecialized.hpp']),
 ('cutlass/epilogue/fusion/sm90_visitor_store_tma_warpspecialized.hpp',
  ['cutlass/epilogue/fusion/ppu_visitor_store_tma_warpspecialized.hpp']),
 ('cutlass/epilogue/fusion/sm90_visitor_compute_tma_warpspecialized.hpp',
  ['cutlass/epilogue/fusion/ppu_visitor_compute_tma_warpspecialized.hpp']),
 ('cutlass/gemm/collective/sm80_mma_multistage.hpp', ['cutlass/gemm/collective/ppu_mma_multistage.hpp']),
 ('cutlass/gemm/collective/sm70_mma_twostage_ldmatrix.hpp',
  ['cutlass/gemm/collective/ppu_mma_twostage_ldmatrix.hpp']),
 ('cutlass/gemm/kernel/sm70_gemm.hpp', ['cutlass/gemm/kernel/ppu_gemm.hpp']),
 ('cutlass/gemm/kernel/sm90_tile_scheduler.hpp', ['cutlass/gemm/kernel/ppu_tile_scheduler.hpp']),
 ('cutlass/gemm/kernel/sm90_tile_scheduler_group.hpp', ['cutlass/gemm/kernel/ppu_tile_scheduler_group.hpp']),
 ('cutlass/gemm/kernel/sm90_tile_scheduler_stream_k.hpp',
  ['cutlass/gemm/kernel/ppu_tile_scheduler_stream_k.hpp']),
 ('cutlass/gemm/threadblock/default_mma_core_sm80.h', ['cutlass/gemm/threadblock/default_mma_core_ppu.h']),
 ('cutlass/gemm/thread/mma_sm50.h', ['cutlass/gemm/thread/mma_ppu.h']),
 ('cutlass/gemm/thread/mma_sm60.h', ['cutlass/gemm/thread/mma_ppu.h']),
 ('cutlass/gemm/thread/mma_sm61.h', ['cutlass/gemm/thread/mma_ppu.h'])]

TOKEN_RENAMES = {'cu_count': 'sm_count',
 'device_cus_': 'device_sms_',
 'cu_occupancy_': 'sm_occupancy_',
 'device_cus': 'device_sms',
 'cu_occupancy': 'sm_occupancy',
 'avail_cus_': 'avail_sms_',
 'avail_cus': 'avail_sms',
 'available_cu_count': 'available_sm_count',
 'override_cu_count': 'override_sm_count',
 'max_cu_per_gpc': 'max_sm_per_gpc',
 'tb_per_cu': 'tb_per_sm',
 'MaxCuCount': 'MaxSmCount',
 'CuArch': 'SmArch',
 'dest_cu': 'dest_sm',
 'PtgType': 'NvType',
 'PtgTypeV2': 'NvTypeV2'}

ARCH_HEADER = 'cutlass/arch/hggc_arch_map.h'

ARCH_CONTENT = ('#pragma once\n'
 '// HGGC <-> CUDA architecture macro mapping\n'
 '// The PPU compiler auto-defines __HGGC_ARCH__ (100/150); CUDA/nvcc does not.\n'
 '// This header defines __HGGC_ARCH__ based on __CUDA_ARCH__.\n'
 '// Strict mapping: HGGC 100 <-> CUDA 800 (Sm80), HGGC 150 <-> CUDA 890 (Sm89)\n'
 '// If __HGGC_ARCH__ is already externally defined, validate it matches __CUDA_ARCH__; error on mismatch.\n'
 '\n'
 '#if defined(__CUDA_ARCH__)\n'
 '#  if __CUDA_ARCH__ == 800\n'
 '#    if defined(__HGGC_ARCH__)\n'
 '#      if __HGGC_ARCH__ != 100\n'
 '#        error "__HGGC_ARCH__ mismatch: expected 100 for __CUDA_ARCH__ 800"\n'
 '#      endif\n'
 '#    else\n'
 '#      define __HGGC_ARCH__ 100\n'
 '#    endif\n'
 '#  elif __CUDA_ARCH__ == 890\n'
 '#    if defined(__HGGC_ARCH__)\n'
 '#      if __HGGC_ARCH__ != 150\n'
 '#        error "__HGGC_ARCH__ mismatch: expected 150 for __CUDA_ARCH__ 890"\n'
 '#      endif\n'
 '#    else\n'
 '#      define __HGGC_ARCH__ 150\n'
 '#    endif\n'
 '#  endif\n'
 '#endif\n')

REQUIRED = ['cutlass/arch/arch.h',
 'cute/config.hpp',
 'cute/arch/copy_ppu.hpp',
 'cutlass/arch/memory_ppu.h',
 'cutlass/ppu_host_adapter.hpp']
