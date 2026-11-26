"""MoonEP source adaptations applied only by the moonep command."""

from general import convert_text


def convert_source(relative, original):
    if relative == 'setup.py':
        # ltOki availability depends on the HGCC version in both modes.
        start = original.index('def detect_hggc_version()')
        end = original.index('\n\nHGGC_MAJOR,', start)
        text = convert_text(original[:start]) + original[start:end] + convert_text(original[end:])
        if 'libraries.append("pccl")' not in text:
            # PCCL's private dispatch API is exported by libpccl.
            text = text.replace('ext_modules = [\n',
                                'if os.path.isfile(os.path.join(NCCL_INCLUDE_DIR, "nccl_device", "kernel_dispatch.h")):\n'
                                '    library_dirs.append(os.path.normpath(os.path.join(os.path.dirname(NCCL_LIBRARY), "..", "..", "lib")))\n'
                                '    libraries.append("pccl")\n\n\n'
                                'ext_modules = [\n')
        return text

    text = convert_text(original)
    if relative == 'csrc/jit/compiler.hpp' and 'get_hgcc_version()' in original:
        text = text.replace('        return {major, minor};\n',
                            '        MOONEP_HOST_ASSERT((major > 12 or (major == 12 and minor >= 3)) and "NVCC >= 12.3 required");\n'
                            '        return {major, minor};\n')
    return text
