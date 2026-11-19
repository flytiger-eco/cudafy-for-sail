"""DeepEP source adaptations applied only by the deepep command."""

import re

from general import convert_text


def convert_source(relative, original):
    return _apply_source_adaptations(relative, original, convert_text(original))


def _convert_extension_config(text):
    # NCCL's GIN minimum uses its own version encoding, not a PCCL release mapping.
    text = text.replace('MIN_GIN_NCCL_VERSION_CODE = 2200',
                        'MIN_GIN_NCCL_VERSION_CODE = 23004')
    text = text.replace("return f'{code // 1000}.{code % 1000 // 100}.{code % 100}'",
                        "scale = 1000 if code < 10000 else 10000\n"
                        "    return f'{code // scale}.{code % scale // 100}.{code % 100}'")
    text = text.replace('return major * 1000 + minor * 100 + patch, header',
                        'scale = 1000 if major <= 2 and minor <= 8 else 10000\n'
                        '    return major * scale + minor * 100 + patch, header')
    if "libraries=['pccl'] if has_kernel_dispatch else []," not in text:
        text = text.replace('    extra_compile_args = {\n',
                            "    has_kernel_dispatch = need_pccl and (Path(pccl_root_dir) / 'include/nccl_device/kernel_dispatch.h').is_file()\n"
                            "    if has_kernel_dispatch:\n"
                            "        library_dirs.append(str(Path(pccl_root_dir).parent / 'lib'))\n"
                            '    extra_compile_args = {\n')
        # PCCL's private dispatch API is exported by libpccl.
        text = text.replace("                name='deep_ep._C',\n",
                            "                name='deep_ep._C',\n"
                            "                libraries=['pccl'] if has_kernel_dispatch else [],\n")
    return text


def _convert_jit_compiler(text):
    text = text.replace('        return {major, minor};\n',
                        '        EP_HOST_ASSERT((major > 12 or (major == 12 and minor >= 3)) and "NVCC version should be >= 12.3");\n'
                        '        return {major, minor};\n')
    return text


def _apply_source_adaptations(relative, original, text):
    if relative == 'deep_ep/utils/find_pkgs.py':
        text = text.replace("find_pkg_root('pccl',", "find_pkg_root('nccl',")
        text = text.replace("find_pkg_root('sailshmem',", "find_pkg_root('nvshmem',")
        text = text.replace("('nvcc', ('PPU_SDK', 'PPU_PATH'), '/usr/local/PPU_SDK')",
                            "('nvcc', ('CUDA_HOME', 'CUDA_PATH'), '/usr/local/cuda')")
    if relative == 'deep_ep/__init__.py':
        text = text.replace("if 'pccl' in line", "if 'libnccl.so' in line")
    if relative == 'csrc/jit/compiler.hpp' and 'HGCCCompiler' in original:
        text = _convert_jit_compiler(text)
    if relative == 'deep_ep/buffers/legacy.py' and 'torch.version.cuda.split' not in text:
        text = re.sub(r"(?m)^([ \t]*)if use_mxfp4:\n([ \t]+)(?=\S)",
                      lambda m: m[1] + "if use_mxfp4:\n" + m[2] +
                      "assert tuple(map(int, torch.version.cuda.split('.'))) >= (12, 8)\n" + m[2], text)
    if relative == 'setup.py':
        text = _convert_extension_config(text)
    if relative == 'tmake.build':
        text = text.replace('export NVSHMEM_DIR=${OUTPUT_DIR}/sailSHMEM',
                            'export NVSHMEM_DIR=${NVSHMEM_DIR:-${OUTPUT_DIR}/sailSHMEM/nvshmem_wrapper}')
    return text
