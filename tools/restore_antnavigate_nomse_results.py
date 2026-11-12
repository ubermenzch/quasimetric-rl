#!/usr/bin/env python3
"""Restore three completed NoMSE runs and discard their duplicate attempts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import qrl_queue_control as control
from tools import run_qrl_queue as queue


LEGACY_ROOT = Path('/home/zhangcheng/qrl-assets/results/queue')
EXPECTED = {'1000': 0.935, '1001': 0.965, '1002': 0.960}
PREFIX = 'iclr_rebuttal_v1_AQRL0-NoMSE-L_500000steps_antnavigate_v4_online_s'


@dataclass
class Recovery:
    task: queue.Task
    status: dict[str, str]
    source: Path
    destination: Path
    test_success: float


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def tree_hashes(root: Path) -> dict[str, str]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f'不是普通结果目录: {root}')
    result = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError(f'结果目录含符号链接，拒绝自动处理: {path}')
        if path.is_file():
            result[str(path.relative_to(root))] = _file_hash(path)
        elif not path.is_dir():
            raise ValueError(f'结果目录含非普通文件: {path}')
    return result


def validate_source(task: queue.Task, source: Path) -> float:
    if (not (source / 'COMPLETE').is_file()
            or not queue.output_finished_for_task(source, task)
            or queue.task_manifest_fingerprint(source) != queue.task_fingerprint(task)
            or not queue.final_evaluation_complete(task, source)):
        raise ValueError(f'旧结果不完整或任务指纹不符: {source}')
    records = [json.loads(line) for line in (source / 'test.log').read_text().splitlines() if line.strip()]
    if len(records) != 1:
        raise ValueError(f'旧结果的测试记录数量不为 1: {source}')
    test = records[0]
    selection = json.loads((source / 'best_checkpoint.json').read_text())
    expected = EXPECTED[task.seed]
    if (test.get('split') != 'test'
            or int(test.get('completed_training_env_steps', -1)) != 500000
            or int(test.get('num_episodes', -1)) != 200
            or int(test.get('seed', -1)) != 1500
            or abs(float(test.get('succ_rate', -1)) - expected) > 1e-9
            or selection.get('test') != test):
        raise ValueError(f'旧结果的训练步数、测试设置或结果不符: {source}')
    return float(test['succ_rate'])


def process_matches(pid: str, task_id: str) -> None:
    if not pid.isdigit() or int(pid) <= 1:
        raise ValueError(f'进程号无效: {pid!r}')
    proc = Path('/proc') / pid
    try:
        if proc.stat().st_uid != os.geteuid():
            raise ValueError(f'进程 {pid} 不属于当前用户')
        args = (proc / 'cmdline').read_bytes().split(b'\0')
        if f'output_folder={task_id}'.encode() not in args:
            raise ValueError(f'进程 {pid} 与任务 {task_id} 不匹配')
        os.kill(int(pid), 0)
    except ProcessLookupError as exc:
        raise RuntimeError(
            f'无法向进程 {pid} 发信号；请在服务器终端运行此脚本'
        ) from exc


def _is_running(pid: str) -> bool:
    return queue.plateau_pid_is_running(pid)


def terminate_duplicates(plans: list[Recovery]) -> None:
    for item in plans:
        process_matches(item.status['pid'], item.task.task_id)
    for item in plans:
        os.kill(int(item.status['pid']), signal.SIGTERM)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if all(not _is_running(item.status['pid']) for item in plans):
            return
        time.sleep(0.2)
    remaining = [item for item in plans if _is_running(item.status['pid'])]
    for item in remaining:
        process_matches(item.status['pid'], item.task.task_id)
        os.kill(int(item.status['pid']), signal.SIGKILL)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if all(not _is_running(item.status['pid']) for item in plans):
            return
        time.sleep(0.2)
    raise RuntimeError('重复训练进程未能全部停止')


def make_plan(config: dict[str, str], legacy_root: Path) -> tuple[list[Recovery], Path, Path]:
    tasks_path = queue.resolve_path(queue.cfg(config, 'TASKS_FILE', 'runs/qrl_queue/tasks.tsv'))
    status_dir = queue.resolve_path(queue.cfg(config, 'STATUS_DIR', 'runs/qrl_queue/status'))
    target_root = queue.results_root(config)
    legacy_root = legacy_root.resolve()
    if legacy_root == target_root or not legacy_root.is_dir() or not target_root.is_dir():
        raise ValueError('旧结果目录或当前结果目录不正确')
    tasks = queue.read_tasks(tasks_path)
    plans = []
    for seed in EXPECTED:
        task_id = PREFIX + seed
        matches = [task for task in tasks if task.task_id == task_id]
        if len(matches) != 1:
            raise ValueError(f'队列中没有唯一的任务: {task_id}')
        task = matches[0]
        if task.env_name != 'AntNavigate-v4' or task.seed != seed or task.steps != '500000':
            raise ValueError(f'任务配置不符: {task_id}')
        status = queue.read_status(status_dir, task_id)
        source, destination = legacy_root / task_id, target_root / task_id
        success = validate_source(task, source)
        if (status.get('state') != 'RUNNING'
                or status.get('task_fingerprint') != queue.task_fingerprint(task)
                or Path(status.get('output_dir', '')).resolve() != destination.resolve()
                or not destination.is_dir() or destination.is_symlink()
                or queue.task_manifest_fingerprint(destination) != queue.task_fingerprint(task)
                or queue.output_finished_for_task(destination, task)):
            raise ValueError(f'重复任务的状态或结果目录不符合预期: {task_id}')
        process_matches(status.get('pid', ''), task_id)
        plans.append(Recovery(task, status, source, destination, success))
    lock_value = os.environ.get('QRL_QUEUE_LOCK_FILE') or queue.cfg(
        config, 'LOCK_FILE', 'runs/qrl_queue/qrl_queue.lock')
    return plans, status_dir, queue.resolve_path(lock_value)


def _original_time(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')


def restore(config_path: Path, legacy_root: Path = LEGACY_ROOT) -> None:
    config = queue.parse_config(config_path)
    plans, status_dir, lock_path = make_plan(config, legacy_root)
    if not control.lock_is_held(lock_path):
        raise RuntimeError('调度器当前没有运行；为避免改变现有状态，未执行恢复')
    target_root = queue.results_root(config)
    required = sum(p.stat().st_size for item in plans for p in item.source.rglob('*') if p.is_file())
    if shutil.disk_usage(target_root).free < required + 16 * 1024 * 1024:
        raise OSError('当前结果目录剩余空间不足以暂存完整结果')

    stage_root = Path(tempfile.mkdtemp(prefix='.restore-ant-nomse-', dir=target_root))
    scheduler_stopped = False
    locked = False
    installed: list[Recovery] = []
    moved: list[Recovery] = []
    saved_status = {}
    committed = False
    preserve_stage = False
    guard = queue.QueueLock(lock_path)
    try:
        for item in plans:
            source_hashes = tree_hashes(item.source)
            staged = stage_root / item.task.task_id
            shutil.copytree(item.source, staged)
            if tree_hashes(staged) != source_hashes or tree_hashes(item.source) != source_hashes:
                raise RuntimeError(f'复制校验失败: {item.task.task_id}')
            validate_source(item.task, staged)
        print('已暂存并校验三个旧结果，准备停止重复训练。', flush=True)

        control.stop(lock_path, with_jobs=False)
        scheduler_stopped = True
        if not guard.acquire():
            raise RuntimeError('无法取得调度器锁，未停止训练进程')
        locked = True

        for item in plans:
            current = queue.read_status(status_dir, item.task.task_id)
            if (current.get('state') != 'RUNNING'
                    or current.get('pid') != item.status.get('pid')
                    or current.get('task_fingerprint') != queue.task_fingerprint(item.task)
                    or Path(current.get('output_dir', '')).resolve() != item.destination.resolve()
                    or queue.output_finished_for_task(item.destination, item.task)):
                raise RuntimeError(f'任务状态已变化，停止恢复: {item.task.task_id}')
            process_matches(current['pid'], item.task.task_id)
            saved_status[item.task.task_id] = (status_dir / (item.task.task_id + '.status')).read_bytes()
        terminate_duplicates(plans)

        for item in plans:
            partial = stage_root / ('duplicate-' + item.task.seed)
            os.replace(item.destination, partial)
            moved.append(item)
            os.replace(stage_root / item.task.task_id, item.destination)
            installed.append(item)
            validate_source(item.task, item.destination)

        for item in plans:
            original_started = _original_time(item.source / '.qrl_task.json')
            queue.write_status(status_dir, item.task, 'DONE', {
                'exit_code': '0',
                'output_dir': str(item.destination),
                'log_file': str(item.destination / 'output.log'),
                'completion_evidence': 'COMPLETE',
                'completed_from_output': '1',
                'restored_from': str(item.source),
                'duplicate_previous_pid': item.status['pid'],
                'started_at': original_started,
                'gpu_started_at': original_started,
                'finished_at': _original_time(item.source / 'COMPLETE'),
            })
            status = queue.read_status(status_dir, item.task.task_id)
            if status.get('state') != 'DONE' or status.get('output_dir') != str(item.destination):
                raise RuntimeError(f'完成状态校验失败: {item.task.task_id}')

        committed = True
        for item in plans:
            partial = stage_root / ('duplicate-' + item.task.seed)
            try:
                shutil.rmtree(partial)
            except OSError as exc:
                preserve_stage = True
                print(f'无法删除部分重复结果，请手动检查 {partial}: {exc}', file=sys.stderr)
            print(f"种子 {item.task.seed}: DONE，测试成功率 {100 * item.test_success:.2f}%", flush=True)
        guard.release()
        locked = False
        if control.start(config_path, []) != 0 or not control.lock_is_held(lock_path):
            raise RuntimeError('结果已恢复，但调度器未保持运行')
        scheduler_stopped = False
        print('三个完整结果已恢复；调度器已启动，其他训练进程未被停止。', flush=True)
        if preserve_stage:
            print(f'部分重复结果尚未删除: {stage_root}', file=sys.stderr)
    except BaseException:
        if locked and not committed:
            for item in reversed(moved):
                partial = stage_root / ('duplicate-' + item.task.seed)
                try:
                    if item in installed and item.destination.exists() and partial.exists():
                        shutil.rmtree(item.destination)
                    if partial.exists() and not item.destination.exists():
                        os.replace(partial, item.destination)
                except OSError as exc:
                    preserve_stage = True
                    print(f'恢复原目录时出错，请检查 {item.destination}: {exc}', file=sys.stderr)
            for item in plans:
                try:
                    if item.task.task_id not in saved_status:
                        continue
                    if _is_running(item.status['pid']):
                        (status_dir / (item.task.task_id + '.status')).write_bytes(
                            saved_status[item.task.task_id])
                    else:
                        queue.write_status(status_dir, item.task, 'PAUSED', {
                            'error': 'restore_incomplete_manual_intervention',
                            'previous_pid': item.status['pid'],
                            'output_dir': str(item.destination),
                        })
                except OSError as exc:
                    preserve_stage = True
                    print(f'恢复任务状态时出错，请检查 {item.task.task_id}: {exc}', file=sys.stderr)
        if scheduler_stopped:
            print('请检查调度器状态；其他训练进程未被此脚本停止。', file=sys.stderr)
        if preserve_stage:
            print(f'保留临时目录供检查: {stage_root}', file=sys.stderr)
        raise
    finally:
        if locked:
            guard.release()
        if stage_root.exists() and not preserve_stage:
            shutil.rmtree(stage_root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/qrl_queue.env')
    args = parser.parse_args()
    try:
        restore(args.config.resolve())
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f'恢复失败: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
