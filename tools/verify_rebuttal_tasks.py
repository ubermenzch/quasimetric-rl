#!/usr/bin/env python3
"""Run short, isolated end-to-end checks using the actual queued configurations."""
from dataclasses import replace
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_qrl_queue import build_command, parse_config, read_tasks, task_uses_native_qrl
from tools.watch_qrl_queue import task_variant
from tools.generate_selected_success_l_min32_tasks import extra_arg_map

HORIZONS = {'FetchPush': 50, 'FetchReach': 50, 'FetchReachImage': 50,
            'FetchPushImage': 50, 'FetchSlide': 50, 'FetchPickAndPlace': 50,
            'reacher_hard': 1000, 'maze2d-large': 800, 'Pusher-v4': 100,
            'AntMaze_UMaze-v5': 700, 'PandaFlip-v3': 50, 'PointMaze_Medium-v3': 600}
HORIZONS.update({'AntNavigate-v4': 1000, 'PandaPickAndPlaceJoints-v3': 50,
                 'PandaPushJoints-v3': 50, 'PandaSlide-v3': 50,
                 'PandaPickAndPlace-v3': 50})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--output', type=Path, default=Path('/tmp/aqrl_rebuttal_preflight'))
    parser.add_argument('--indices', type=int, nargs='*')
    parser.add_argument('--timeout', type=int, default=240)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    tasks = read_tasks(ROOT / 'runs/qrl_queue/tasks.tsv')
    runtime_config = parse_config(ROOT / 'configs/qrl_queue.env')
    selected = []
    seen = set()
    for index, task in enumerate(tasks, 1):
        if not task.task_id.startswith('iclr_rebuttal_v1_'):
            continue
        if args.indices and index not in args.indices:
            continue
        key = (task_variant(task), task.env_name)
        if key in seen:
            continue
        seen.add(key)
        selected.append((index, task))
    args.output.mkdir(parents=True, exist_ok=True)
    failures = []
    for index, task in selected:
        length = HORIZONS[task.env_name]
        values = extra_arg_map(task)
        # Only shorten the interaction/collection/evaluation schedule; preserve
        # real backbones, loss functions, search steps, and action distributions.
        values.update({'env.init_num_transitions': str(length * 2),
                       'env.increment_num_transitions': str(length * 2),
                       'interaction.num_prefill_episodes': '1',
                       'interaction.num_rollouts_per_cycle': '1',
                       'interaction.num_samples_per_cycle': '1',
                       'interaction.num_eval_episodes': '2',
                       'batch_size': str(args.batch_size), 'log_steps': '1', 'save_steps': str(length),
                       'eval_steps': str(length)})
        if not task_uses_native_qrl(task):
            values.update({'logs_only': 'true', 'resume_if_possible': 'false',
                           'interaction.num_test_episodes': '2',
                           'save_replay_buffer': 'false', 'save_final_replay_buffer': 'false'})
        else:
            values['queue.logs_only'] = 'true'
        smoke = replace(task, task_id=f'smoke_{index}_{int(time.time())}', steps=str(length * 2),
                        extra_args=' '.join(f'{k}={v}' for k, v in values.items()))
        cmd, env, cwd, output = build_command({
            **runtime_config,
            'RESULTS_ROOT': str(args.output), 'RESUME_IF_POSSIBLE': '0',
            'ONLINE_SAVE_REPLAY_BUFFER': 'False', 'CPU_THREADS_PER_TASK': '2',
        }, smoke, args.gpu)
        log = args.output / f'{smoke.task_id}.log'
        started = time.monotonic()
        try:
            with log.open('w') as stream:
                result = subprocess.run(cmd, cwd=cwd, env=env, stdout=stream,
                                        stderr=subprocess.STDOUT, timeout=args.timeout)
            ok = result.returncode == 0 and (output / 'COMPLETE').exists()
            if ok and list(output.glob('*.pth')):
                ok = False
            if ok and not task_uses_native_qrl(task):
                ok = all((output / name).is_file() for name in ('eval.log', 'test.log', 'best_checkpoint.json'))
            if ok and (values.get('collect_rebuttal_diagnostics') == 'true' or task_uses_native_qrl(task)):
                summary_path = output / 'diagnostics/run_summary.json'
                ok = summary_path.is_file() and json.loads(summary_path.read_text())['state'] == 'complete'
                ok = ok and bool(list((output / 'diagnostics/evaluations').glob('*.npz')))
            if ok and values.get('agent.actor.losses.min_dist.latent_goal_probe_replay') == 'true':
                probes = list((output / 'diagnostics/probes').glob('completion_*.npz'))
                ok = bool(probes)
                if ok:
                    with np.load(probes[0], allow_pickle=False) as probe:
                        ok = probe['search_h'].shape[0] == 33 and probe['search_cost'].shape[0] == 33
                        ok = ok and len(probe['reference_states']) > 0 and np.isfinite(probe['reference_latents']).all()
            if ok and values.get('agent.actor.losses.min_dist.replay_goal_log_diagnostics') == 'true':
                probes = list((output / 'diagnostics/probes').glob('replay_selection_*.npz'))
                ok = bool(probes)
                if ok:
                    with np.load(probes[0], allow_pickle=False) as probe:
                        ok = probe['candidate_states'].shape[-2] == 31 and len(probe['selected_state']) > 0
            print(f'{"OK" if ok else "FAIL"} {index} {task_variant(task)} {task.env_name} '
                  f'{time.monotonic()-started:.1f}s log={log}', flush=True)
            if not ok:
                failures.append(index)
                print('\n'.join(log.read_text(errors='replace').splitlines()[-18:]), flush=True)
        except subprocess.TimeoutExpired:
            failures.append(index)
            print(f'TIMEOUT {index} log={log}', flush=True)
    print(f'Checked {len(selected)} configurations; failed: {failures}', flush=True)
    return bool(failures)


if __name__ == '__main__':
    raise SystemExit(main())
