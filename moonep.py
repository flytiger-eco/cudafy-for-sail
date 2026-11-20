"""MoonEP source adaptations applied only by the moonep command."""

from general import convert_text


def convert_source(relative, original):
    # PCCL's host dispatch API still takes HGfunction in CUDA-compatible builds.
    if relative == 'csrc/dispatch_strategy.cc':
        return original

    if relative == 'setup.py':
        # ltOki availability depends on the HGCC version in both modes.
        start = original.index('def detect_hggc_version()')
        end = original.index('\n\nHGGC_MAJOR,', start)
        return convert_text(original[:start]) + original[start:end] + convert_text(original[end:])

    text = convert_text(original)
    if relative == 'csrc/jit/compiler.hpp' and 'get_hgcc_version()' in original:
        text = text.replace('        return {major, minor};\n',
                            '        MOONEP_HOST_ASSERT((major > 12 or (major == 12 and minor >= 3)) and "NVCC >= 12.3 required");\n'
                            '        return {major, minor};\n')
    return text
