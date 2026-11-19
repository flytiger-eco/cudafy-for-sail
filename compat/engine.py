"""在 SDK 转换的虚拟文件视图上生成版本限定的 ACTLIZE 兼容产物。"""
import importlib
import os
import re

from general import _extra_roots, _within, convert_text

CODE_EXTENSIONS = {'.h', '.hpp', '.cuh', '.cu', '.c', '.cpp', '.cc', '.cxx', '.inl', '.inc'}
OWNER = '// cudafy-compat: actlize:'


def _profile(version):
    module = {'0.8.0': 'actlize_v0_8', '1.0.0': 'actlize_v1_0'}.get(version)
    if module is None:
        raise ValueError(f'没有 ACTLIZE {version} 的库兼容 profile')
    return importlib.import_module(f'compat.{module}')


def _normal_path(root, relative):
    if os.path.isabs(relative) or '..' in relative.split('/'):
        raise ValueError(f'规则路径越界：{relative}')
    path = os.path.join(root, *[convert_text(part) for part in relative.split('/')])
    if os.path.realpath(path) != path:
        raise ValueError(f'规则路径包含 symlink：{path}')
    return path


def _token_converter(mapping):
    pattern = re.compile(r'\b(?:' + '|'.join(re.escape(k) for k in sorted(mapping, key=len, reverse=True)) + r')\b')
    return lambda text: pattern.sub(lambda match: mapping[match.group(0)], text)


def _without_comments(text):
    return re.sub(r'/\*.*?\*/|//[^\n]*', '', text, flags=re.S)


def _namespace_at(text, index):
    # 仅识别词法 namespace/块边界；字符串内的花括号不是作用域。
    text = re.sub(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
                  lambda match: ' ' * len(match.group(0)), text[:index])
    stack = []
    for match in re.finditer(r'\b(?:inline\s+)?namespace(?:\s+([\w:]+))?\s*\{|[{}]', text):
        token = match.group(0)
        if token == '}':
            if stack:
                stack.pop()
        elif token == '{':
            stack.append(('<block>',))
        else:
            stack.append(tuple((match.group(1) or '<anonymous>').split('::')))
    return '::'.join(part for scope in stack for part in scope)


_ALIAS_RE = re.compile(r'(?P<template>\btemplate\s*<[^;{}]+?>\s*)?'
                       r'\busing\s+(?P<name>\w+)\s*=(?P<target>[^;]+);')


def _append_aliases(content, namespace, body, marker, path):
    begin, end = marker + ' aliases begin', marker + ' aliases end'
    block = f'{begin}\nnamespace {namespace} {{\n{body.rstrip()}\n}}\n{end}\n'
    existing_block = re.search(re.escape(begin) + r'\n.*?' + re.escape(end) + r'\n',
                               content, flags=re.S)
    if begin in content and (existing_block is None or content.count(begin) != 1):
        raise ValueError(f'已有别名块与 profile 不一致：{path}')
    # 已经存在的真实别名不重复声明；不一致或不能确认的定义拒绝覆盖。
    base = content if existing_block is None else content[:existing_block.start()] + content[existing_block.end():]
    clean = _without_comments(base)
    declarations = {}
    for existing in _ALIAS_RE.finditer(clean):
        if _namespace_at(clean, existing.start()) == namespace:
            declarations.setdefault(existing.group('name'), []).append(existing)
    removals = []
    for match in _ALIAS_RE.finditer(body):
        name = match.group('name')
        existing = declarations.get(name, [])
        signature = lambda item: tuple(re.sub(r'\s+', '', item.group(key) or '')
                                       for key in ['template', 'target'])
        if existing:
            if any(signature(item) != signature(match) for item in existing):
                raise ValueError(f'别名定义冲突：{path}: {namespace}::{name}')
            removals.append((match.start(), match.end()))
        else:
            for occupied in re.finditer(r'\b(?:struct|class|enum)\s+' + re.escape(name) + r'\b', clean):
                if _namespace_at(clean, occupied.start()) == namespace:
                    raise ValueError(f'兼容名已被真实类型占用：{path}: {name}')
    for start, stop in reversed(removals):
        body = body[:start] + body[stop:]
    if re.search(r'\bint\s+SmId\s*\(', body):
        for occupied in re.finditer(r'\bint\s+SmId\s*\(', clean):
            if _namespace_at(clean, occupied.start()) == namespace:
                raise ValueError(f'SmId 定义冲突：{path}')
    block = f'{begin}\nnamespace {namespace} {{\n{body.rstrip()}\n}}\n{end}\n'
    if existing_block is not None:
        if existing_block.group(0) != block:
            raise ValueError(f'已有别名块与 profile 不一致：{path}')
        return content
    guard = re.search(r'(?m)^\s*#ifndef\s+(\w+)\s*\n\s*#define\s+\1\b', content)
    if guard:
        closing = list(re.finditer(r'(?m)^\s*#endif\b[^\n]*', content))
        if closing:
            index = closing[-1].start()
            return content[:index] + '\n' + block + content[index:]
    return content.rstrip('\n') + '\n\n' + block


def _insert_include(content, include):
    if include in content:
        return content
    pragma = re.search(r'(?m)^[ \t]*#\s*pragma\s+once[^\n]*(?:\n|$)', content)
    guard = re.search(r'(?m)^[ \t]*#ifndef\s+(\w+)[ \t]*\n[ \t]*#define\s+\1[^\n]*(?:\n|$)', content)
    match = pragma or guard
    index = match.end() if match else 0
    return content[:index] + include + '\n' + content[index:]


def extend_actlize_plan(plan, include_root, version):
    """扩展只读计划；缺失结构和冲突在任何写盘之前报错。"""
    profile = _profile(version)
    root = os.path.abspath(include_root)
    marker = OWNER + version
    roots = [root] + _extra_roots(root)
    for relative in profile.REQUIRED:
        path = _normal_path(root, relative)
        if path not in plan.files:
            raise ValueError(f'ACTLIZE {version} 必需结构缺失：{relative}')
    for path, text in plan.files.items():
        if not any(_within(path, scope) for scope in roots):
            continue
        for existing in re.findall(re.escape(OWNER) + r'([^\s]+)', text):
            if existing != version:
                raise ValueError(f'异版本生成产物：{path}: {existing} != {version}')
    convert_tokens = _token_converter(profile.TOKEN_RENAMES)
    for path in list(plan.files):
        if (any(_within(path, scope) for scope in roots)
                and os.path.splitext(path)[1] in CODE_EXTENSIONS):
            plan.files[path] = convert_tokens(plan.files[path])
    for relative, (namespace, body) in profile.ALIASES.items():
        path = _normal_path(root, relative)
        if path not in plan.files:
            plan.diagnostics.append(f'可选别名目标缺失，跳过：{relative}')
            continue
        body = convert_tokens(convert_text(body))
        plan.files[path] = _append_aliases(plan.files[path], namespace, body, marker, path)
    wrapper_edges = {}
    specifications = {}
    for relative, includes in profile.WRAPPERS:
        path = _normal_path(root, relative)
        targets = [_normal_path(root, inc) for inc in includes]
        if path in specifications and specifications[path] != targets:
            raise ValueError(f'wrapper 规则路径冲突：{relative}')
        specifications[path] = targets
        if path in targets:
            # SDK 已经把真实实现重命名到此入口，不能覆盖成自包含 wrapper。
            if path not in plan.files:
                raise ValueError(f'自包含 wrapper：{relative}')
            continue
        if any(target not in plan.files for target in targets):
            plan.diagnostics.append(f'可选 wrapper 实现缺失，跳过：{relative}')
            continue
        body = '#pragma once\n' + marker + ' wrapper\n'
        body += ''.join(f'#include <{os.path.relpath(target, root)}>\n' for target in targets)
        if path in plan.files:
            if OWNER in plan.files[path] and plan.files[path] != body:
                raise ValueError(f'已有 wrapper 与 profile 不一致：{relative}')
            if OWNER in plan.files[path]:
                wrapper_edges[path] = targets
            # 真实入口文件保留原实现，不覆盖。
            continue
        plan.add_file(path, body)
        wrapper_edges[path] = targets
    def visit(path, stack):
        if path in stack:
            raise ValueError(f'wrapper 包含循环：{path}')
        for target in wrapper_edges.get(path, ()):
            visit(target, stack | {path})
    for path in wrapper_edges:
        visit(path, set())
    arch_path = _normal_path(root, profile.ARCH_HEADER)
    arch_content = marker + ' architecture\n' + convert_text(profile.ARCH_CONTENT)
    if arch_path in plan.files and plan.files[arch_path] != arch_content:
        raise ValueError(f'已有架构桥接与 profile 不一致：{arch_path}')
    if arch_path not in plan.files:
        plan.add_file(arch_path, arch_content)
    include = f'#include "{os.path.relpath(arch_path, root)}"'
    for path, text in list(plan.files.items()):
        if (path != arch_path and any(_within(path, scope) for scope in roots)
                and os.path.splitext(path)[1] in CODE_EXTENSIONS
                and re.search(r'\b__HGGC_ARCH__\b', _without_comments(text))):
            plan.files[path] = _insert_include(text, include)
    return plan
