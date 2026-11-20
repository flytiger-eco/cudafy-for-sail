"""DeepEP source adaptations applied only by the deepep command."""

import re

from general import convert_text


def convert_source(relative, original):
    return _apply_source_adaptations(relative, original, convert_text(original))


def _convert_extension_config(text):
    text = text.replace("include_dirs.append(f'{pccl_root_dir}/include')",
                        "include_dirs.append(f'{pccl_root_dir}/include_cuda')")
    text = text.replace("include_dirs.append(f'{sailshmem_dir}/include')",
                        "include_dirs.append(f'{sailshmem_dir}/include_cuda')")
    text = text.replace("'vendored' / 'sailshmem' / 'include'",
                        "'vendored' / 'sailshmem' / 'include_cuda'")
    for native_arch, cuda_arch in (('ppu_10;ppu_15', '8.0'), ('ppu_10', '8.0a'), ('ppu_15', '8.9')):
        text = text.replace(f"'{native_arch}'", f"'{cuda_arch}'")
    text = text.replace('expected one of ppu_10;ppu_15, ppu_10, or ppu_15',
                        'expected one of 8.0, 8.0a, or 8.9')
    return text


def _convert_jit_compiler(text):
    text = text.replace('        int major, minor;\n',
                        '        // The version should be at least 12.3\n'
                        '        int major, minor;\n')
    text = text.replace('        return {major, minor};\n',
                        '        EP_HOST_ASSERT((major > 12 or (major == 12 and minor >= 3)) and "NVCC version should be >= 12.3");\n'
                        '        return {major, minor};\n')
    text = text.replace('flags += fmt::format(" -I {}/include", pccl_root.c_str());',
                        'flags += fmt::format(" -I {}/include_cuda", pccl_root.c_str());')
    text = text.replace('flags += fmt::format(" -I{}/include", sailshmem_dir);',
                        'flags += fmt::format(" -I{}/include_cuda", sailshmem_dir);')
    return text


def _apply_source_adaptations(relative, original, text):
    if relative == 'deep_ep/utils/find_pkgs.py':
        text = text.replace("('nvcc', ('PPU_SDK', 'PPU_PATH'), '/usr/local/PPU_SDK')",
                            "('nvcc', ('CUDA_HOME', 'CUDA_PATH'), '/usr/local/cuda')")
    if relative == 'deep_ep/__init__.py':
        text = text.replace("os.path.join(_vendored_sailshmem, 'include')",
                            "os.path.join(_vendored_sailshmem, 'include_cuda')")
    if relative == 'csrc/jit/compiler.hpp' and 'HGCCCompiler' in original:
        text = _convert_jit_compiler(text)
    if relative == 'deep_ep/buffers/legacy.py' and 'torch.version.cuda.split' not in text:
        text = re.sub(r"(?m)^([ \t]*)if use_mxfp4:\n([ \t]+)(?=\S)",
                      lambda m: m[1] + "if use_mxfp4:\n" + m[2] +
                      "assert tuple(map(int, torch.version.cuda.split('.'))) >= (12, 8)\n" + m[2], text)
    if relative in ('csrc/kernels/legacy/internode_ll_direct.cu',
                    'deep_ep/include/deep_ep/legacy/impls/internode_ll_defs.cuh') and 'CUDA_VERSION >= 12080' not in text:
        text = re.sub(r'(__device__ __forceinline__ __nv_fp8_storage_t cvt_float2_to_fp4x2\(.*?\n\})',
                      r'#if defined(CUDA_VERSION) && CUDA_VERSION >= 12080\n\1\n#endif', text, flags=re.DOTALL)
        if 'void mxfp4_quantize_phase1(' in text:
            start = text.index('void mxfp4_quantize_phase1(')
            text = text[:start] + text[start:].replace(
                '#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)',
                '#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 890) && defined(CUDA_VERSION) && CUDA_VERSION >= 12080', 1)
    if relative == 'deep_ep/include/deep_ep/legacy/impls/internode_ll_dispatch.cuh' and 'CUDA_VERSION >= 12080' not in text:
        text = re.sub(r'(template<>\n__device__ __forceinline__ void quant_process<false, false, true,.*?\n\})',
                      r'#if defined(CUDA_VERSION) && CUDA_VERSION >= 12080\n\1\n#endif', text, flags=re.DOTALL)
    if relative == 'setup.py' and 'CppExtension(' in text and 'hgcc_dlink' in original:
        text = _convert_extension_config(text)
    if relative == 'install.sh':
        text = text.replace('local sdk_home="${PPU_SDK:-$PPU_PATH}"',
                            'local sdk_home="${CUDA_HOME:-$CUDA_PATH}"')
        text = text.replace('"$SAILSHMEM_DIR/include" "$vendored_dir/include"',
                            '"$SAILSHMEM_DIR/include_cuda" "$vendored_dir/include_cuda"')
    return text
