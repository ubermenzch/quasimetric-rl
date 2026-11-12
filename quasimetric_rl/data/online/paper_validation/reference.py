"""Load pinned, locally available upstream environments without their RL stacks."""

from __future__ import annotations

import ast
import contextlib
import importlib
import os
from pathlib import Path
import subprocess
import sys
import threading
import types


REVISIONS = {
    'gcsl-official': 'cfae5609cee79e5a2228fb7653451023c41a64cb',
    'google-research-gcrl': 'ec7c3d346277b737bc2decffcd1b533d4b7ec105',
    'td-infonce-official': '18f4e7e5872da9c3653f57661d01b4fbce85b50e',
}
METAWORLD_REVISION = '33f3b90495be99f02a61da501d7d661e6bc675c5'
CRL_METAWORLD_REVISION = 'a0009ed9a208ff9864a5c1368c04c273bb20dd06'
_MAZE_CONSTRUCTION_LOCK = threading.RLock()
_METAWORLD_IMPORT_LOCK = threading.RLock()


def make_ant_maze(ant_module, **kwargs):
    """Release D4RL's discarded mkstemp descriptor and generated model XML."""
    from d4rl.locomotion import maze_env

    with _MAZE_CONSTRUCTION_LOCK:
        original_tempfile = maze_env.tempfile
        paths = []

        def closed_mkstemp(*args, **options):
            descriptor, path = original_tempfile.mkstemp(*args, **options)
            paths.append(Path(path))
            os.close(descriptor)
            return descriptor, path

        # D4RL writes by filename and discards the returned descriptor. Limit
        # the compatibility shim to its module, leaving global tempfile alone.
        maze_env.tempfile = types.SimpleNamespace(mkstemp=closed_mkstemp)
        try:
            return ant_module.AntMaze(**kwargs)
        finally:
            maze_env.tempfile = original_tempfile
            for path in paths:
                path.unlink(missing_ok=True)


def reference_root() -> Path:
    return Path(os.environ.get(
        'QRL_PAPER_REFERENCE_ROOT', '/home/zhangcheng/gcrl-reference-repos',
    ))


def repository(name: str) -> Path:
    path = reference_root() / name
    if not path.is_dir():
        raise FileNotFoundError(f'Missing pinned reference repository: {path}')
    revision = subprocess.check_output(
        ['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True,
    ).strip()
    if revision != REVISIONS[name]:
        raise RuntimeError(f'{path}: expected {REVISIONS[name]}, got {revision}')
    changes = subprocess.check_output(
        ['git', '-C', str(path), 'status', '--porcelain', '--untracked-files=no'],
        text=True,
    )
    if changes.strip():
        raise RuntimeError(f'Reference source has tracked modifications: {path}')
    return path


def namespace(name: str, path: Path) -> None:
    """Skip upstream __init__ files that eagerly import unused simulators."""
    if name in sys.modules:
        if str(path) not in getattr(sys.modules[name], '__path__', ()):
            raise RuntimeError(f'Conflicting already-imported package: {name}')
        return
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    module.__package__ = name
    sys.modules[name] = module


class LazyPlot:
    def __getattr__(self, name):
        return getattr(importlib.import_module('matplotlib.pyplot'), name)


def load_module(name: str, path: Path, *, skip_imports=(), skip_definitions=(),
                strip_gin=False):
    """Only strip dependency plumbing; do not rewrite environment method bodies."""
    if name in sys.modules:
        return sys.modules[name]
    tree = ast.parse(path.read_text(), filename=str(path))
    kept = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or '']
        else:
            names = []
        if any(n == p or n.startswith(p + '.') for n in names for p in skip_imports):
            continue
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in skip_definitions:
            continue
        kept.append(node)
    tree.body = kept
    if strip_gin:
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                node.decorator_list = [
                    deco for deco in node.decorator_list
                    if not ast.unparse(deco).startswith('gin.configurable')
                ]
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = name.rpartition('.')[0]
    module.plt = LazyPlot()
    sys.modules[name] = module
    try:
        exec(compile(tree, str(path), 'exec'), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def gcsl_dependencies() -> Path:
    path = repository('gcsl-official')
    deps = path / 'dependencies'
    namespace('multiworld', deps / 'multiworld')
    namespace('room_world', deps / 'room_world')
    load_module('room_world.rooms', deps / 'room_world/rooms.py',
                skip_imports=('matplotlib',))
    # Explicit preload must also expose the child on its parent package.
    sys.modules['room_world'].rooms = sys.modules['room_world.rooms']
    return path


def gcsl_native_env(name):
    path = gcsl_dependencies()
    namespace('gcsl', path / 'gcsl')
    namespace('gcsl.envs', path / 'gcsl/envs')
    module, cls = {
        'GCSLLunarLander': ('lunarlander', 'LunarEnv'),
        'GCSLClaw': ('claw_env', 'ClawEnv'),
    }[name]
    return getattr(importlib.import_module('gcsl.envs.' + module), cls)()


@contextlib.contextmanager
def crl_metaworld_imports():
    """Load CRL classes without replacing C-Learning's imported MetaWorld."""
    path = Path(os.environ.get(
        'QRL_CRL_METAWORLD', str(reference_root() / 'metaworld-crl'),
    ))
    revision = subprocess.check_output(
        ['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True,
    ).strip()
    if revision != CRL_METAWORLD_REVISION:
        raise RuntimeError(f'Wrong CRL MetaWorld revision: {revision}')
    with _METAWORLD_IMPORT_LOCK:
        previous = {name: module for name, module in sys.modules.copy().items()
                    if name == 'metaworld' or name.startswith('metaworld.')}
        for name in previous:
            del sys.modules[name]
        sys.path.insert(0, str(path))
        try:
            yield importlib.import_module('metaworld')
        finally:
            sys.path.remove(str(path))
            for name in list(sys.modules):
                if name == 'metaworld' or name.startswith('metaworld.'):
                    del sys.modules[name]
            sys.modules.update(previous)


def crl_sawyer_env(name):
    source = repository('google-research-gcrl')
    # All upstream MetaWorld imports are module-level; the returned classes
    # retain their own module globals after the temporary import context exits.
    with crl_metaworld_imports():
        module = load_module(
            '_qrl_crl_sawyer', source / 'contrastive_rl/env_utils.py',
            skip_imports=('ant_env', 'fetch_envs', 'point_env'),
        )
        return getattr(module, {'CRLSawyerPush': 'SawyerPush',
                                'CRLSawyerBin': 'SawyerBin'}[name])()


def require_metaworld() -> Path:
    path = Path(os.environ.get(
        'QRL_C_LEARNING_METAWORLD', str(reference_root() / 'metaworld-c-learning'),
    ))
    if not (path / 'metaworld').is_dir():
        raise FileNotFoundError(
            f'C-Learning needs MetaWorld {METAWORLD_REVISION} at {path}. '
            'Set QRL_C_LEARNING_METAWORLD to the matching source checkout; '
            'modern MetaWorld/Gymnasium tasks are not interchangeable.'
        )
    revision = subprocess.check_output(
        ['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True,
    ).strip()
    if revision != METAWORLD_REVISION:
        raise RuntimeError(f'Wrong C-Learning MetaWorld revision: {revision}')
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    # Instantiate original classes directly; old Gym registration uses removed tags.
    namespace('metaworld', path / 'metaworld')
    namespace('metaworld.envs', path / 'metaworld/envs')
    namespace('metaworld.envs.mujoco', path / 'metaworld/envs/mujoco')
    return path
