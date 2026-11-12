#!/usr/bin/env python3
"""Run original QRL unchanged, omitting model checkpoint writes only."""
import logging
import inspect
from pathlib import Path
import runpy
import subprocess
import sys
import time

import torch
try:
    from .rebuttal_diagnostics import RunDiagnostics
    from .git_metadata import git_output
except ImportError:
    from rebuttal_diagnostics import RunDiagnostics
    from git_metadata import git_output


def is_model_checkpoint(path):
    if not isinstance(path, (str, Path)):
        return False
    name = Path(path).name
    return name.startswith(('checkpoint_', 'agent_checkpoint_')) and name.endswith('.pth')


def run_training():
    original_save = torch.save
    arguments = dict(token.split('=', 1) for token in sys.argv[1:] if '=' in token)
    output = (Path(arguments['output_base_dir']) / arguments['output_folder']
              if 'output_base_dir' in arguments and 'output_folder' in arguments else None)
    diagnostics = RunDiagnostics(output, protocol={'arguments': sys.argv[1:], 'source': str(Path.cwd())}) if output else None
    original_train = original_eval = trainer_type = None
    base_conf_type = original_setup = None
    writers = []
    last = {'env_steps': 0, 'optim_steps': 0}
    if diagnostics is not None:
        from online.trainer import Trainer
        from quasimetric_rl.base_conf import BaseConf
        base_conf_type, original_setup = BaseConf, BaseConf.setup_for_experiment

        def setup_for_experiment(self):
            writer = original_setup(self)
            writers.append(writer)
            return writer

        BaseConf.setup_for_experiment = setup_for_experiment
        trainer_type = Trainer
        original_train, original_eval = Trainer.train_step, Trainer.evaluate

        def train_step(self, *args, **kwargs):
            context = inspect.currentframe().f_back.f_locals
            last.update(env_steps=int(context.get('env_steps', 0)),
                        optim_steps=int(context.get('optim_steps', 0)))
            started = time.perf_counter()
            if last['optim_steps'] == 1:
                diagnostics.record_runtime(self)
            try:
                result = original_train(self, *args, **kwargs)
            except Exception as exc:
                data = args[0] if args else kwargs.get('data')
                if data is not None and hasattr(type(data), '__attrs_attrs__'):
                    arrays = {field.name: getattr(data, field.name)
                              for field in type(data).__attrs_attrs__
                              if isinstance(getattr(data, field.name), torch.Tensor)}
                    diagnostics.record_probe('failure_batch', last['optim_steps'], arrays,
                                             env_steps=last['env_steps'], metadata={'error': str(exc)})
                raise
            if last['optim_steps'] == 1 or last['optim_steps'] % 250 == 0:
                diagnostics.record_training(last['env_steps'], last['optim_steps'], result,
                                            iteration_seconds=time.perf_counter() - started,
                                            data=args[0] if args else kwargs.get('data'), agent=self.agent)
            return result

        def evaluate(self, *args, **kwargs):
            context = inspect.currentframe().f_back.f_locals
            result = original_eval(self, *args, **kwargs)
            diagnostics.record_evaluation(result, dict(
                split='validation', env_steps=int(context.get('env_steps', 0)),
                optim_steps=int(context.get('optim_steps', 0)),
                seed=getattr(self, 'eval_seed', None),
            ))
            return result

        Trainer.train_step, Trainer.evaluate = train_step, evaluate

    def metrics_only_save(obj, path, *args, **kwargs):
        if is_model_checkpoint(path):
            for writer in writers:
                writer.flush()
            logging.info('Checkpoint persistence disabled (logs-only task): %s', path)
            return None
        return original_save(obj, path, *args, **kwargs)

    torch.save = metrics_only_save
    state, error = 'failed', None
    try:
        runpy.run_module('online.main', run_name='__main__')
        state = 'complete' if output is not None and (output / 'COMPLETE').exists() else 'config_only'
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        if isinstance(exc, SystemExit) and not exc.code:
            state = 'config_only'
        raise
    finally:
        torch.save = original_save
        if trainer_type is not None:
            trainer_type.train_step, trainer_type.evaluate = original_train, original_eval
        if base_conf_type is not None:
            base_conf_type.setup_for_experiment = original_setup
        for writer in writers:
            writer.flush()
            writer.close()
        if diagnostics is not None:
            diagnostics.finish(state, error=error, **last)


def main():
    original_check_output = subprocess.check_output

    def metadata_output(command, **kwargs):
        supported = (["rev-parse", "--show-toplevel"], ["rev-parse", "HEAD"],
                     ["status", "--short"], ["status", "--porcelain", "--untracked-files=no"])
        if (isinstance(command, (list, tuple)) and command and command[0] == "git"
                and list(command[1:]) in supported
                and set(kwargs) <= {"cwd", "text", "encoding", "universal_newlines"}):
            value = git_output(command[1:], cwd=kwargs.get("cwd"))
            if kwargs.get("text") or kwargs.get("encoding") or kwargs.get("universal_newlines"):
                return value
            return value.encode()
        return original_check_output(command, **kwargs)

    subprocess.check_output = metadata_output
    try:
        return run_training()
    finally:
        subprocess.check_output = original_check_output


if __name__ == '__main__':
    main()
