#!/usr/bin/env python3
"""DeepGEMM conversion -- general engine + DeepGEMM-specific residuals.

The general engine (full SDK naming map + compile-chain rules) covers all the
token renames (hggc/hg/acblas families), the hgcc -> nvcc identifier chain,
the arch-flag unification (single compute_80/sm_80 gencode) and the
smxx_acblaslt.hpp -> smxx_cublaslt.hpp rename. What cannot be expressed as
token renames stays here; this file is the COMPLETE residual rule set:

  1. csrc/python_api.cpp -> .cu rename (host TU becomes a CUDA TU);
  2. guard-region removal for `#if defined(__HGGC__)` device-only code that a
     host g++ TU must not see (utils_rtc.cuh / scheduler_cutlass3.cuh);
  3. the JIT compiler rewrites: nvcc discovery, version query, flag-set
     translation (expt-* flags, -Xcompiler merging), RTC include paths and
     the NVRTC version assertion;
  4. setup.py / __init__.py SDK-root plumbing (PPU_SDK -> torch CUDA_HOME);
  5. per-file include fixes (ATen/cuda/CUDAContext.h) and profiling params;
  6. CMakeLists.txt restoration.

Usage (unchanged):
    python3 cuda_compat.py [REPO_DIR] [--dry-run] [--verbose]
"""

import argparse
import os
import shutil
import sys
import tempfile

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from general import convert_path  # noqa: E402

# =============================================================================
# Level 1: file operations
# =============================================================================

FILE_RENAMES = [
    ('csrc/python_api.cpp', 'csrc/python_api.cu'),
]

# =============================================================================
# Level 2: regex replacements (file-scoped)
# =============================================================================

REGEX_REPLACEMENTS = [
    # --- csrc/jit_kernels/impls/fp4_gemm.hpp: expand launch_kernel ---
    (r'DG_CUDA_UNIFIED_CHECK\(launch_kernel\(kernel, configs, args\.kernel_params\)\);',
     'void* ptr_args[] = {(void*)&args.kernel_params};\n        DG_CUDA_UNIFIED_CHECK(lazy_cuLaunchKernelEx(&configs, kernel, ptr_args, nullptr));',
     'fp4_gemm.hpp'),

    # --- utils_rtc.cuh / scheduler_cutlass3.cuh: drop the __HGGC__ guards ---
    # The compat nvcc predefines __HGGC__ for device code but a host g++ TU
    # does not, and the guarded bodies are device-only; the guard is removed
    # so the product keeps the code. The body pattern is line-bounded and
    # refuses to cross other preprocessor directives, so regions can never
    # be mis-paired.
    (r'(?m)^#if defined\(__HGGC__\)\n((?:(?!^#\s*(?:if|ifdef|ifndef|else|elif|endif)\b)[^\n]*\n)*)^#endif  // __HGGC__\n',
     r'\1', 'utils_rtc.cuh'),
    (r'(?m)^#if defined\(__HGGC__\)\n((?:(?!^#\s*(?:if|ifdef|ifndef|else|elif|endif)\b)[^\n]*\n)*)^#endif  // defined\(__HGGC__\)\n',
     r'\1', 'scheduler_cutlass3.cuh'),

    # --- profiling_interface.cuh: the runtime include comes from torch ---
    (r'#include <cuda_runtime.h>\n', '', 'profiling_interface.cuh'),
]

# =============================================================================
# Level 3: file-specific text replacements (anchored on POST-GENERAL text)
# =============================================================================

FILE_SPECIFIC_REPLACEMENTS = {
    'csrc/jit/compiler.hpp': [
        # Includes (the general engine already renamed hggc_runtime_api.h /
        # hggc.h; the compat build wants ATen + cuda_runtime.h instead).
        ('#include <cuda_runtime_api.h>', '#include <ATen/cuda/CUDAContext.h>'),
        ('#include <cuda.h>\n', '#include <cuda_runtime.h>\n'),
        # The offline PPU compiler rejects --diag-suppress; re-inject it.
        ('flags = fmt::format("-std=c++{} ",',
         'flags = fmt::format("-std=c++{} --diag-suppress=39,174,177,940 ",'),
        # ptxas resource report, gated on the debug envs (anchored on the
        # 8-space LINEINFO if-line; the RtcOptions copy uses 12 spaces).
        ('        if (get_env("DG_JIT_WITH_LINEINFO", 0))\n',
         '        if (get_env("DG_JIT_DEBUG", 0) or get_env("DG_JIT_PTXAS_VERBOSE", 0) or get_env("DG_JIT_PTXAS_CHECK", 0))\n            flags += " --ptxas-options=--verbose,--warn-on-local-memory-usage";\n        if (get_env("DG_JIT_WITH_LINEINFO", 0))\n'),
        # NVCC ctor signature uses the (major, minor) pair.
        ('        signature = fmt::format("NVCC{}", get_nvcc_version());',
         '        const auto& [nvcc_major, nvcc_minor] = get_nvcc_version();\n        signature = fmt::format("NVCC{}.{}", nvcc_major, nvcc_minor);'),
        # Flag-set translation (PPU offline flags -> nvcc equivalents).
        ('flags += "-cubin -ftemplate-depth=8192 -O3 -DNDEBUG ";',
         'flags += "-cubin --expt-relaxed-constexpr --expt-extended-lambda ";'),
        ('flags += "-Xcompiler -fPIC ";', 'flags += "";'),
        ('flags += "-Xcompiler -Wno-deprecated-declarations -Xcompiler -Wno-abi ";',
         'flags += "-Xcompiler -O3,-Wno-deprecated-declarations,-Wno-abi ";'),
        # RTC options: re-add the __CUDACC__ define at the slot where the
        # general engine dropped -DUSE_HGGC.
        ('            // "-U__linux__",\n\n            };',
         '            // "-U__linux__",\n            // "-D__CUDACC_RTC__",\n            "-D__CUDACC__",\n            };'),
        # RTC include lists: the CUDA side needs the extra toolkit paths.
        ('std::string sdk_include = std::string(getenv("PPU_HOME")) + "/include";',
         'std::string cuda_home = std::string(getenv("PPU_HOME")) + "/include";'),
        ('std::string sdk_include_cccl = std::string(getenv("PPU_HOME")) + "/include/cccl";',
         'std::string cuda_home1 = std::string(getenv("PPU_HOME")) + "/include/cccl";'),
        ('includes_insert({sdk_include, sdk_include_cccl});',
         'includes_insert({cuda_home, cuda_home1, cuda_home1 + "/cuda/std"});'),
        ('includes_insert({sdk_include});',
         'includes_insert({cuda_home, cuda_home + "/cuda/std", cuda_home + "/../targets/x86_64-linux/include/thrust/system/cuda"});'),
        # NVRTC signature + version assertion.
        ('        signature = fmt::format("HGRTC{}.{}", major, minor);',
         '        signature = fmt::format("NVRTC{}.{}", major, minor);\n        DG_HOST_ASSERT((major > 12 or (major == 12 and minor >= 3)) and "NVRTC version should be >= 12.3");'),
    ],

    'csrc/jit/device_runtime.hpp': [
        ('#include <cuda_runtime_api.h>', '#include <cuda_runtime.h>'),
        ('#include <torch/torch.h>', '#include <ATen/cuda/CUDAContext.h>'),
    ],

    'csrc/utils/utils.hpp': [
        ('#include <cuda_runtime_api.h>', '#include <ATen/cuda/CUDAContext.h>'),
    ],

    'deep_gemm/include/deep_gemm/impls/w4a16_gemm_cutlass3.cuh': [
        ('#pragma clang diagnostic push\n#pragma clang diagnostic ignored "-Wunknown-attributes"',
         '#pragma clang diagnostic push\n#pragma clang diagnostic ignored "-Wunknown-attributes"\n#pragma clang diagnostic ignored "-Wcuda-compat"'),
    ],

    'setup.py': [
        # SDK root block: replaced by torch's CUDA_HOME (import added below).
        ("ppu_sdk = os.environ.get('PPU_SDK', '/usr/local/PPU_SDK')\n"
         "ppu_include = os.path.join(ppu_sdk, 'targets', 'x86_64-linux', 'include')\n\n",
         ""),
        ("# PPU SDK path — provides cuda headers for actlize\n", ""),
        ("from torch.utils.cpp_extension import CppExtension, BuildExtension",
         "from torch.utils.cpp_extension import CppExtension, CUDA_HOME, CUDAExtension, BuildExtension"),
        ("sources = ['csrc/python_api.cpp']", "sources = ['csrc/python_api.cu']"),
        ("    ppu_include,", "    f'{CUDA_HOME}/include',"),
        ("build_libraries = ['cuda', 'hggcrt1', 'hgrtc', 'cublasLt']",
         "build_libraries = ['cuda', 'cudart', 'nvrtc', 'cublasLt']"),
        ("    os.path.join(ppu_sdk, 'lib'),",
         "    f'{CUDA_HOME}/lib64',\n    f'{CUDA_HOME}/lib64/stub'"),
        # Extension type.
        ("CppExtension(name='deep_gemm.deep_gemm_cpp',",
         "CUDAExtension(name='deep_gemm.deep_gemm_cpp',"),
    ],

    'deep_gemm/__init__.py': [
        # The SDK root derivation becomes a torch CUDA_HOME import. It must
        # run here (post-general) because the original line still reads
        # PPU_SDK / PPU_HOME.
        ("# SDK root: env PPU_SDK > env PPU_HOME > default\nPPU_HOME = os.environ.get('PPU_SDK') or os.environ.get('PPU_HOME') or '/usr/local/PPU_SDK'",
         'from torch.version import cuda as cuda_version\nfrom packaging import version\nfrom torch.utils.cpp_extension import CUDA_HOME'),
        # deep_gemm_cpp.init() takes the SDK root as a parameter.
        ('    PPU_HOME         # SDK root', '    CUDA_HOME         # SDK root'),
    ],

    'deep_gemm/jit/compiler.py': [
        ('import torch\nfrom typing import Tuple',
         'import torch\nfrom torch.utils.cpp_extension import CUDA_HOME\nfrom typing import Tuple'),
        # Flag-set translation.
        ("        '-shared',                  # produce a .so rather than a raw binary\n\n    ]",
         "        '-shared',                  # produce a .so rather than a raw binary\n"
         "        '--expt-relaxed-constexpr', '--expt-extended-lambda', '--diag-suppress=39,174,177,940',\n    ]"),
        ("nvcc_flags.extend(['-ftemplate-depth=8192', '-O3', '-DNDEBUG'])",
         "nvcc_flags.extend(['-O3'])"),
        ("nvcc_flags.extend(['-Xcompiler', '-fPIC', '-Xcompiler', '-Wno-deprecated-declarations', '-Xcompiler', '-Wno-abi'])",
         "nvcc_flags.extend(['--compiler-options=-fPIC,-O3,-Wno-deprecated-declarations,-Wno-abi'])"),
    ],

    'deep_gemm/jit/interleave_ffma.py': [
        ("PPU_HOME = os.environ.get('PPU_SDK') or os.environ.get('PPU_HOME') or '/usr/local/PPU_SDK'",
         'from torch.utils.cpp_extension import CUDA_HOME'),
        ('{PPU_HOME}/bin/cuobjdump', '{CUDA_HOME}/bin/cuobjdump'),
    ],

    'csrc/jit_kernels/impls/m_grouped_int8_gemm.hpp': [
        ('expected_m, layout_info,', 'expected_m, m_rows_tensor.data_ptr<int32_t>(),'),
        ('        return std::make_pair(block_m, ceil_div(n, block_n));', '        return;'),
    ],
}

# =============================================================================
# Level 4: structural region replacements (anchored on POST-GENERAL text)
# =============================================================================

FILE_SPECIFIC_BLOCK_REPLACEMENTS = {
    'deep_gemm/jit/compiler.py': [
        # The compiler-discovery helper is a genuine rewrite (CUDA_HOME
        # fallback, version assertions); anchored on the post-general def
        # line, the interior may be edited freely.
        ('get_compiler_fn',
         r'def get_nvcc_compiler\(\) -> Tuple\[str, str\]:\n.*?(?=\n\n@functools\.lru_cache\(maxsize=None\)\ndef get_default_user_dir\(\):)',
         r"""def get_nvcc_compiler() -> Tuple[str, str]:
    paths = []
    if os.getenv('DG_NVCC_COMPILER'):
        paths.append(os.getenv('DG_NVCC_COMPILER'))
    paths.append(f'{CUDA_HOME}/bin/nvcc')

    # Try to find the first available NVCC compiler
    least_version_required = '11.6'
    version_pattern = re.compile(r'release (\d+\.\d+)')
    for path in paths:
        if os.path.exists(path):
            match = version_pattern.search(os.popen(f'{path} --version').read())
            version = match.group(1)
            assert match, f'Cannot get the version of NVCC compiler {path}'
            assert version >= least_version_required, f'NVCC {path} version {version} is lower than {least_version_required}'
            return path, version
    raise RuntimeError('Cannot find any available NVCC compiler')"""),
    ],
    'csrc/jit/compiler.hpp': [
        # The version query becomes a (major, minor) pair; the PPU source
        # carries porting-landmark comments around the method, which the
        # leading/trailing comment-tolerant pattern absorbs.
        ('version',
         r'(?:    //[^\n]*\n)*    std::string get_nvcc_version\(\) const \{\n.*?\n    \}\n(?:    //[^\n]*\n)*',
         r"""    std::pair<int, int> get_nvcc_version() const {
        DG_HOST_ASSERT(std::filesystem::exists(nvcc_path));

        // Call the version command
        const auto& command = std::string(nvcc_path) + " --version";
        const auto& [return_code, output] = call_external_command(command);
        DG_HOST_ASSERT(return_code == 0);

        // The version should be at least 12.3, for the best performance with 12.9
        int major, minor;
        std::smatch match;
        DG_HOST_ASSERT(std::regex_search(output, match, std::regex(R"(release (\d+\.\d+))")));
        std::sscanf(match[1].str().c_str(), "%d.%d", &major, &minor);
        DG_HOST_ASSERT((major > 12 or (major == 12 and minor >= 3)) and "NVCC version should be >= 12.3");
        if (major == 12 and minor < 9)
            printf("Warning: please use at least NVCC 12.9 for the best DeepGEMM performance\n");
        return {major, minor};
    }
"""),
    ],
}

# =============================================================================
# Level 5: post-conversion assertions (post-general + residual expectations)
# =============================================================================

REQUIRED_AFTER_CONVERSION = {
    'csrc/jit/compiler.hpp': [
        '#include <ATen/cuda/CUDAContext.h>',
        '#include <cuda_runtime.h>',
        '#include <nvrtc.h>',
        'std::pair<int, int> get_nvcc_version() const {',
        'nvcc_path = sdk_home / "bin" / "nvcc";',
        'get_env<std::string>("DG_JIT_NVCC_COMPILER")',
        'const auto& [nvcc_major, nvcc_minor] = get_nvcc_version();',
        'signature = fmt::format("NVCC{}.{}", nvcc_major, nvcc_minor);',
        '-gencode=arch=compute_80,code=sm_80',
        '-cubin --expt-relaxed-constexpr --expt-extended-lambda',
        '--diag-suppress=39,174,177,940',
        'if (get_env("DG_JIT_DEBUG", 0) or get_env("DG_JIT_PTXAS_VERBOSE", 0) or get_env("DG_JIT_PTXAS_CHECK", 0))',
        '-Xcompiler -O3,-Wno-deprecated-declarations,-Wno-abi',
        'includes_insert({cuda_home, cuda_home1, cuda_home1 + "/cuda/std"});',
        'thrust/system/cuda',
        'signature = fmt::format("NVRTC{}.{}", major, minor);',
        'and "NVRTC version should be >= 12.3"',
        'kernel.cubin',
        'nvcc_path',
        '"-D__CUDACC__",',
    ],
    'deep_gemm/jit/compiler.py': [
        'from torch.utils.cpp_extension import CUDA_HOME',
        'def get_nvcc_compiler() -> Tuple[str, str]:',
        "paths.append(f'{CUDA_HOME}/bin/nvcc')",
        "least_version_required = '11.6'",
        "'--expt-relaxed-constexpr', '--expt-extended-lambda', '--diag-suppress=39,174,177,940',",
        "'--compiler-options=-fPIC,-O3,-Wno-deprecated-declarations,-Wno-abi'",
        "'-gencode=arch=compute_80,code=sm_80'",
        'nvcc_flags',
        'nvcc.tmp.',
    ],
    'csrc/jit/device_runtime.hpp': [
        '#include <ATen/cuda/CUDAContext.h>',
        'std::shared_ptr<cudaDeviceProp> cached_prop;',
    ],
    'setup.py': [
        'from torch.utils.cpp_extension import CppExtension, CUDA_HOME, CUDAExtension, BuildExtension',
        "sources = ['csrc/python_api.cu']",
        "f'{CUDA_HOME}/include',",
        "build_libraries = ['cuda', 'cudart', 'nvrtc'",
        "f'{CUDA_HOME}/lib64',",
        "f'{CUDA_HOME}/lib64/stub'",
        "CUDAExtension(name='deep_gemm.deep_gemm_cpp',",
    ],
    'deep_gemm/__init__.py': [
        'from torch.version import cuda as cuda_version',
        'from packaging import version',
        'from torch.utils.cpp_extension import CUDA_HOME',
        'deep_gemm_cpp.init(',
        'CUDA_HOME         #',
    ],
}


# =============================================================================
# Engine
# =============================================================================

def apply_replacements(content, replacements):
    for old, new in replacements:
        if old and old != new:
            if old in new and new in content:
                continue  # idempotency guard for insertion-type rules
            content = content.replace(old, new)
    return content


def apply_regex_replacements(content, regex_list, filepath):
    import re
    for pattern, replacement, file_glob in regex_list:
        if file_glob and not filepath.endswith(file_glob):
            continue
        content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    return content


def apply_block_replacements(content, blocks, rel_path):
    import re
    for name, pattern, cuda_text in blocks:
        rx = re.compile(pattern, re.DOTALL)
        new_content, n = rx.subn(lambda m: cuda_text, content)
        if n == 0:
            if cuda_text.strip() and cuda_text in content:
                continue  # already converted
            raise RuntimeError(
                "[deepgemm residual] Structural region '%s' not found in %s. "
                "Its landmark lines were altered or removed; refusing to "
                "emit a half-converted file." % (name, rel_path))
        if n > 1:
            raise RuntimeError(
                "[deepgemm residual] Structural region '%s' matched %d times "
                "in %s (ambiguous landmark)." % (name, n, rel_path))
        content = new_content
    return content


def verify_required(content, rel_path):
    required = REQUIRED_AFTER_CONVERSION.get(rel_path)
    if not required:
        return
    missing = [s for s in required if s not in content]
    if missing:
        raise RuntimeError(
            "[deepgemm residual] Converted %s is missing expected CUDA text:\n%s\n"
            "  A conversion rule silently failed to match." % (
                rel_path, "\n".join("    missing: %r" % s for s in missing[:20])))


def verify_python_syntax(content, rel_path, original=None):
    if not rel_path.endswith('.py'):
        return
    try:
        compile(content, rel_path, 'exec')
        return
    except SyntaxError as e:
        converted_err = e
    if original is not None:
        try:
            compile(original, rel_path, 'exec')
        except SyntaxError:
            return
    raise RuntimeError(
        "[deepgemm residual] Conversion broke the syntax of %s: %s (line %s)." % (
            rel_path, converted_err.msg, converted_err.lineno))


def handle_file_renames(repo_dir, dry_run=False, verbose=False):
    changes = 0
    for old_rel, new_rel in FILE_RENAMES:
        old_path = os.path.join(repo_dir, old_rel)
        new_path = os.path.join(repo_dir, new_rel)
        if os.path.exists(old_path) and not os.path.exists(new_path):
            if dry_run:
                print(f"  [DRY-RUN] Would rename: {old_rel} -> {new_rel}")
            else:
                os.rename(old_path, new_path)
                print(f"  [RENAMED] {old_rel} -> {new_rel}")
            changes += 1
        elif os.path.exists(old_path) and os.path.exists(new_path):
            if dry_run:
                print(f"  [DRY-RUN] Would remove: {old_rel} (keeping {new_rel})")
            else:
                os.remove(old_path)
                print(f"  [REMOVED] {old_rel} (keeping {new_rel})")
            changes += 1
        elif os.path.exists(new_path) and verbose:
            print(f"  [SKIP] {new_rel} already exists, {old_rel} not found")
    return changes


CMAKE_CONTENT = """\
# NOTES: current just for CMake-based IDE (e.g. CLion) indexing, the real compilation is done via JIT
# TODO: add CUDA utils' library via CMake
cmake_minimum_required(VERSION 3.10)
project(deep_gemm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(CUDAToolkit REQUIRED)
find_package(pybind11 REQUIRED)

file(WRITE ${CMAKE_BINARY_DIR}/test_cuda.cu "extern \\"C\\" __global__ void testKernel() { }")
execute_process(
        COMMAND ${CUDA_NVCC_EXECUTABLE} ${CMAKE_CUDA_FLAGS} -gencode arch=compute_90a,code=sm_90a -o ${CMAKE_BINARY_DIR}/test_cuda.o -c ${CMAKE_BINARY_DIR}/test_cuda.cu
        RESULT_VARIABLE NVCC_RESULT
        OUTPUT_VARIABLE NVCC_OUTPUT
        ERROR_VARIABLE NVCC_ERROR_OUTPUT
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

if (NVCC_RESULT EQUAL "0")
    set(NVCC_SUPPORTS_SM90 TRUE)
    message(STATUS "NVCC supports SM90")
else()
    message(STATUS "NVCC does not support SM90")
endif()

if (NVCC_SUPPORTS_SM90)
    set(TORCH_CUDA_ARCH_LIST "8.6" CACHE STRING "Add arch tag 90a to NVCC" FORCE)
    list(APPEND CUDA_NVCC_FLAGS "-gencode;arch=compute_90a,code=sm_90a")
endif()
find_package(Torch REQUIRED)

include_directories(deep_gemm/include third-party/cutlass/include third-party/cutlass/tools/util/include)
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include ${TORCH_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
link_directories(${TORCH_INSTALL_PREFIX}/lib ${CUDA_TOOLKIT_ROOT_DIR}/lib)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -fPIC -DNDEBUG")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -std=c++17 -DNDEBUG")

cuda_add_library(example_gemm STATIC indexing/main.cu)
"""


def restore_cmakelists(repo_dir, dry_run=False):
    cmake_path = os.path.join(repo_dir, 'CMakeLists.txt')
    if os.path.exists(cmake_path):
        print("  [SKIP] CMakeLists.txt already exists")
        return 0
    if dry_run:
        print("  [DRY-RUN] Would restore CMakeLists.txt")
    else:
        with open(cmake_path, 'w') as f:
            f.write(CMAKE_CONTENT)
        print("  [RESTORED] CMakeLists.txt")
    return 1


def process_file(filepath, repo_dir, dry_run=False, verbose=False):
    rel_path = os.path.relpath(filepath, repo_dir)
    try:
        with open(filepath, 'r', encoding='utf-8', errors='surrogateescape',
                  newline='') as f:
            original = f.read()
    except OSError as exc:
        print(f"  [residual] WARNING: cannot read {rel_path}: {exc}")
        return 0

    content = original
    if rel_path in FILE_SPECIFIC_BLOCK_REPLACEMENTS:
        content = apply_block_replacements(
            content, FILE_SPECIFIC_BLOCK_REPLACEMENTS[rel_path], rel_path)
    if rel_path in FILE_SPECIFIC_REPLACEMENTS:
        content = apply_replacements(content, FILE_SPECIFIC_REPLACEMENTS[rel_path])
    content = apply_regex_replacements(content, REGEX_REPLACEMENTS, filepath)

    verify_required(content, rel_path)
    verify_python_syntax(content, rel_path, original)

    if content == original:
        return 0
    if not dry_run:
        try:
            with open(filepath, 'w', encoding='utf-8', errors='surrogateescape',
                      newline='') as f:
                f.write(content)
        except OSError as exc:
            print(f"  [residual] WARNING: cannot write {rel_path}: {exc}")
            return 0
    if verbose:
        prefix = "[DRY-RUN] residual" if dry_run else "[residual]"
        print(f"  {prefix} {rel_path}")
    return 1


def _preview_tree(repo_dir):
    preview = tempfile.TemporaryDirectory(prefix="cudafy-deepgemm-")
    work_dir = os.path.join(preview.name, "repo")
    shutil.copytree(
        repo_dir, work_dir,
        ignore=shutil.ignore_patterns(".git", "__pycache__", "build", "dist",
                                      ".eggs", "*.egg-info"),
    )
    return preview, work_dir


def main():
    parser = argparse.ArgumentParser(
        description='DeepGEMM PPU-original -> CUDA-Compatible Transform '
                    '(general engine + residuals)')
    parser.add_argument('repo_dir', nargs='?', default='.',
                        help='Path to DeepGemm repository')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without making changes')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed output')
    args = parser.parse_args()

    repo_dir = os.path.abspath(args.repo_dir)
    if not os.path.isdir(repo_dir):
        print(f"ERROR: Repository directory not found: {repo_dir}")
        return 1

    print("=" * 70)
    print("  DeepGemm PPU-original -> CUDA Compatible")
    print(f"  Repository: {repo_dir}")
    print(f"  Mode: {'DRY-RUN' if args.dry_run else 'APPLY'}")
    print("=" * 70)

    preview = None
    work_dir = repo_dir
    if args.dry_run:
        preview, work_dir = _preview_tree(repo_dir)
        print("  Preview runs against an isolated temporary copy.")

    try:
        print("\n[Phase 1] File operations (renames, restores)...")
        total_changes = handle_file_renames(work_dir, False, args.verbose)
        total_changes += restore_cmakelists(work_dir, False)

        print("\n[Phase 2] General engine (full SDK map + compile chain)...")
        scanned, changed, renamed, _ = convert_path(
            work_dir, dry_run=False, verbose=args.verbose)
        print(f"  Files scanned: {scanned}")
        print(f"  Files changed: {changed}")
        print(f"  Renames:       {renamed}")
        total_changes += changed + renamed

        print("\n[Phase 3] DeepGEMM residual rules...")
        files_modified = 0
        for root, dirs, files in os.walk(work_dir):
            dirs[:] = [d for d in dirs if d not in
                       ('.git', '__pycache__', 'build', 'dist', '.eggs')
                       and not d.endswith('.egg-info')]
            for fname in sorted(files):
                filepath = os.path.join(root, fname)
                rel = os.path.relpath(filepath, work_dir)
                if (rel in FILE_SPECIFIC_REPLACEMENTS or
                        rel in FILE_SPECIFIC_BLOCK_REPLACEMENTS or
                        any(filepath.endswith(g) for _, _, g in REGEX_REPLACEMENTS)):
                    if process_file(filepath, work_dir, False, args.verbose):
                        files_modified += 1
        print(f"  Residual files modified: {files_modified}")
        total_changes += files_modified

        print(f"\n{'='*70}")
        print(f"  Summary: total changes ~{total_changes}")
        if args.dry_run:
            print("  (DRY-RUN mode - no source files modified)")
        print(f"{'='*70}")
        return 0
    finally:
        if preview is not None:
            preview.cleanup()


if __name__ == '__main__':
    sys.exit(main())
