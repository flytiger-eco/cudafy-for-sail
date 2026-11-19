"""Opt-in FA2 2.7.2 library symbols for the separately converted ACTLIZE 0.8.

SDK conversion remains in general. These rules restore the historical FA2
copy/MMA names and device architecture conditions, including the wait macro
that must match ACTLIZE's asynchronous-copy enable macro. Standalone extension
packaging and structural source rewrites are deliberately not part of this
profile; PyTorch owns the integrated FA2 build.
"""
import os
import re

from general import _within
from compat.engine import CODE_EXTENSIONS, _normal_path, _token_converter

TOKEN_RENAMES = {
    'PPU_16x8x16_F32F16F16F32_TN': 'SM80_16x8x16_F32F16F16F32_TN',
    'PPU_16x8x16_F32BF16BF16F32_TN': 'SM80_16x8x16_F32BF16BF16F32_TN',
    'PPU_16x8x8_F32F16F16F32_TN': 'SM75_16x8x8_F32F16F16F32_TN',
    'PPU_U32x4_LDSM_N': 'SM75_U32x4_LDSM_N',
    'PPU_U16x8_LDSM_T': 'SM75_U16x8_LDSM_T',
    'PPU_CP_ASYNC_CACHEGLOBAL': 'SM80_CP_ASYNC_CACHEGLOBAL',
    'PPU_CP_ASYNC_CACHEALWAYS': 'SM80_CP_ASYNC_CACHEALWAYS',
    'CUTE_ARCH_CP_ASYNC_PPU_ENABLED': 'CUTE_ARCH_CP_ASYNC_SM80_ENABLED',
    '__HGGC_ARCH__': '__CUDA_ARCH__',
}
REQUIRED = ('setup.py', 'csrc/flash_attn/src/utils.h',
            'csrc/flash_attn/src/kernel_traits.h')
_ARCH_COMPARE = re.compile(r'(\b__CUDA_ARCH__\s*(>=|==)\s*)(100|150)\b')


def convert_library_text(text):
    """Rename whole identifiers; translate only the historical arch comparisons."""
    text = _token_converter(TOKEN_RENAMES)(text)

    def architecture(match):
        value = match.group(3)
        if value == '100':
            return match.group(1) + '800'
        if match.group(2) == '==':
            return match.group(1) + '890'
        return match.group(0)

    return _ARCH_COMPARE.sub(architecture, text)


def extend_flash_attention_plan(plan, target, version):
    """Extend a read-only SDK plan without touching bundled/external libraries."""
    if version != '2.7.2':
        raise ValueError(f'No library compatibility profile for Flash-Attention {version}')
    root = os.path.abspath(target)
    for relative in REQUIRED:
        if _normal_path(root, relative) not in plan.files:
            raise ValueError(f'Flash-Attention {version} required structure missing: {relative}')
    scopes = [os.path.join(root, 'csrc', 'flash_attn'), os.path.join(root, 'hopper')]
    for path, text in list(plan.files.items()):
        if (any(_within(path, scope) for scope in scopes)
                and os.path.splitext(path)[1] in CODE_EXTENSIONS):
            plan.files[path] = convert_library_text(text)
    return plan
