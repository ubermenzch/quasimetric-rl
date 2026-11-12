#!/usr/bin/env python3
"""Preview user caches/old Codex releases; --apply deletes the listed files.

Qwen model downloads are retained unless --qwen-downloads is explicitly given.
Paper sources, experiment artifacts, histories, and model checkpoints are excluded.
"""
import argparse
import os
from pathlib import Path
import re
import shutil
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--apply', action='store_true')
parser.add_argument('--qwen-downloads', action='store_true')
args = parser.parse_args()
user_root = Path('/home/zhangcheng')
owner = user_root.stat().st_uid
if os.getuid() != owner:
    raise SystemExit('Run as zhangcheng, without sudo.')

active_paths = set()
active_qwen = False
for process in Path('/proc').iterdir():
    if not process.name.isdigit() or process.name == str(os.getpid()):
        continue
    try:
        if process.stat().st_uid != owner:
            continue
        command = (process / 'cmdline').read_bytes().split(b'\0')
        active_qwen |= any(b'tf_learning' in part or b'Qwen3' in part for part in command)
        try:
            active_paths.add(os.readlink(process / 'exe'))
        except OSError:
            pass
        for line in (process / 'maps').read_text().splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) == 6 and fields[5].startswith('/'):
                active_paths.add(fields[5].removesuffix(' (deleted)'))
    except OSError:
        continue

candidates = []
cutoff = time.time() - 7 * 86400
for package in ('standalone', 'app-server-daemon'):
    base = user_root / '.codex/packages' / package
    releases = base / 'releases'
    if not releases.is_dir():
        continue
    version = lambda path: tuple(map(int, re.match(r'(\d+)\.(\d+)\.(\d+)', path.name).groups()))
    versions = sorted((p for p in releases.iterdir()
                       if p.is_dir() and not p.is_symlink()
                       and re.match(r'\d+\.\d+\.\d+-', p.name)), key=version, reverse=True)
    keep = set(versions[:2]) | {(base / 'current').resolve()}
    for path in versions:
        in_use = any(value == str(path) or value.startswith(str(path) + '/')
                     for value in active_paths)
        if path not in keep and not in_use and path.stat().st_mtime < cutoff:
            candidates.append(path)

for relative in ('.npm/_cacache', '.cache/pip'):
    path = user_root / relative
    if path.is_dir() and not path.is_symlink():
        candidates.append(path)

if args.qwen_downloads:
    if active_qwen:
        raise SystemExit('A running process references Qwen/tf_learning; model downloads retained.')
    for model in ('Qwen3-0.6B', 'Qwen3-4B'):
        base = user_root / 'tf_learning/models' / model
        for path in base.glob('*.safetensors'):
            if path.is_file() and not path.is_symlink():
                candidates.append(path)

def bytes_used(path):
    if path.is_file():
        return path.stat().st_size
    return sum(p.lstat().st_size for p in path.rglob('*')
               if p.is_file() and not p.is_symlink())

total = 0
for path in candidates:
    if path.is_symlink() or path.stat().st_uid != owner:
        raise SystemExit(f'Refusing unexpected owner or symlink: {path}')
    size = bytes_used(path)
    print(f'{size / 1024**3:.3f} GiB  {path}', flush=True)
    total += size
print(f'Total: {total / 1024**3:.2f} GiB; mode: ' + ('DELETE' if args.apply else 'PREVIEW'), flush=True)
if args.apply:
    for path in candidates:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    print('Cleanup complete. Current and previous Codex releases retained.')
