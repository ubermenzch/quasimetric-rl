#!/usr/bin/env python3
"""Package only the NQRL manuscript's standalone rendering dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "iclr2027"
OUTPUT = ROOT / "nqrl_iclr2027_loonglatex.zip"
TEXT_SUFFIXES = {'.tex', '.bib', '.bst', '.sty', '.md', '.csv', '.json', '.py',
                 '.txt', '.sh', '.yaml', '.yml', '.tsv'}
PRIVATE_PATH = re.compile(r'(?:/home/|/Users/)[\w.-]+|[A-Za-z]:[\\/]+Users[\\/]+[\w.-]+')
PRIVATE_IP = re.compile(r'(?<![\d.])(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}'
                        r'|192\.168\.\d{1,3}\.\d{1,3}'
                        r'|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})(?![\d.])')


def anonymous_bytes(path: Path) -> bytes:
    """Anonymize exported copies; keep original local evidence and its hashes."""
    content = path.read_bytes()
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return content
    text = content.decode('utf-8')
    replacements = {
        str(ROOT): '.',
        str(ROOT.parent / 'qrl-assets/results/queue'): 'artifacts/results/queue',
        str(Path.home()): 'artifacts',
    }
    for source in sorted(replacements, key=len, reverse=True):
        text = text.replace(source, replacements[source])
    text = PRIVATE_PATH.sub('artifacts', text)
    text = PRIVATE_IP.sub('PRIVATE_HOST', text)
    assert not PRIVATE_PATH.search(text) and not PRIVATE_IP.search(text), path.name
    return text.encode('utf-8')


def write_anonymous_archive(output: Path, paths: list[Path], base: Path) -> None:
    """Write portable ZIP entries and a manifest without local account metadata."""
    files, changed = {}, []
    for path in paths:
        name = path.relative_to(base).as_posix()
        data = anonymous_bytes(path)
        files[name] = data
        if data != path.read_bytes():
            changed.append(name)
    report = {
        'description': 'Local account paths and private network addresses are replaced in exported copies. Original local evidence is unchanged.',
        'artifact_paths': 'Paths under artifacts/ identify external source records; these logs are not bundled.',
        'source_hashes': 'Hashes in provenance records identify original source files; archive entry hashes are listed below.',
        'changed_files': changed,
        'sha256': {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
    }
    files['ANONYMIZATION_REPORT.json'] = (json.dumps(report, indent=2) + '\n').encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, 'w', compression=ZIP_DEFLATED) as archive:
        for name, data in files.items():
            entry = ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            entry.compress_type = ZIP_DEFLATED
            entry.create_system = 3
            entry.external_attr = (0o100755 if name.endswith('.sh') else 0o100644) << 16
            archive.writestr(entry, data)
    with ZipFile(output) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(paths) + 1
        for name, expected in report['sha256'].items():
            data = archive.read(name)
            assert hashlib.sha256(data).hexdigest() == expected
            if Path(name).suffix.lower() in TEXT_SUFFIXES:
                text = data.decode('utf-8')
                assert not PRIVATE_PATH.search(text) and not PRIVATE_IP.search(text), name
                assert Path.home().name not in text, f'Local account name remains: {name}'
    print(f'Anonymization verified: {len(changed)} exported text files changed; local originals preserved.')


def rendering_files() -> list[Path]:
    names = {
        "main.tex", "nqrl_iclr2027.tex",
        "nqrl_iclr2027.bib", "math_commands.tex", "iclr2027_conference.sty",
        "iclr2027_conference.bst", "fancyhdr.sty", "natbib.sty",
        "RENDER_README.md", "REVIEW_NOTES.md", "REVISION_NOTES.md",
        "FORMAT_AUDIT.md",
        "data/final_success.csv", "data/goal_distance_steps.csv",
        "data/goal_distance_spread_audit.json",
        "data/goal_distance_steps_runs.csv",
        "data/goal_distance_steps_provenance.json",
        "data/seed_test_success.csv", "data/seed_test_success_audit.json",
        "data/dynamics_mse_summary.csv", "data/dynamics_mse_runs.csv",
        "data/dynamics_validation_runs.csv", "data/dynamics_validation_curves.csv",
        "data/dynamics_validation_audit.json",
        "data/goal_dimension_analysis.csv", "data/goal_dimension_analysis.json",
        "data/goal_fraction_success_runs.csv", "data/goal_fraction_success_tasks.csv",
        "data/goal_fraction_success_seed_means.csv", "data/goal_fraction_success_curves.csv",
        "data/goal_fraction_success_audit.json",
        "data/goal_fraction_gain_runs.csv", "data/goal_fraction_gain_tasks.csv",
        "data/goal_fraction_gain_seed_means.csv", "data/goal_fraction_gain_curves.csv",
        "data/goal_fraction_gain_audit.json",
        "data/dynamics_mse_evidence.csv", "data/dynamics_mse_ant_trajectories.csv",
        "data/dynamics_trajectory_summary.csv", "data/nqrl_parameter_counts.csv",
        "data/dynamics_late_window.csv",
        "data/per_task_auc.csv", "data/per_task_auc_runs.csv",
        "data/per_task_learning_curves.csv", "data/validation_checkpoints.csv",
        "data/per_task_learning_audit.json", "data/online_feedback_runs.csv",
        "data/online_feedback_summary.csv", "data/online_feedback_audit.json",
        "REFERENCE_AUDIT.md", "VISUAL_REFERENCES.md",
        "data/network_architecture_audit.json",
        "data/environment_visualization_audit.json",
        "data/learning_grid_audit.json",
    }
    names.update(str(p.relative_to(PAPER)) for p in (PAPER / "figures/environments").glob("*.png"))
    pending = [PAPER / name for name in sorted(names) if name.endswith(".tex")]
    visited = set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        source = path.read_text(encoding="utf-8")
        source = re.sub(r"(?m)(?<!\\)%.*$", "", source)
        for relative in re.findall(r"\\(?:input|include)\{([^}]+)\}", source):
            if not Path(relative).suffix:
                relative += ".tex"
            names.add(relative)
            pending.append(PAPER / relative)
        names.update(re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", source))
    paths = [PAPER / name for name in sorted(names)]
    for path in paths:
        assert path.is_file(), f"Missing rendering dependency: {path}"
        assert path.resolve().is_relative_to(PAPER.resolve()), f"External dependency: {path}"
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--sources-output', type=Path,
                        help='Also archive figure tools and simulator/model Python sources in repository layout')
    parser.add_argument('--include-pdf', action='store_true',
                        help='Include nqrl_iclr2027.pdf after separately building and verifying it')
    args = parser.parse_args()
    output = args.output
    paths = rendering_files()
    if args.include_pdf:
        pdf = PAPER / 'nqrl_iclr2027.pdf'
        assert pdf.is_file(), 'Build and verify nqrl_iclr2027.pdf before including it'
        audit_path = PAPER / 'data/compilation_audit.json'
        audit = json.loads(audit_path.read_text())
        assert audit['pdf_sha256'] == hashlib.sha256(pdf.read_bytes()).hexdigest()
        assert audit['main_text_pages'] <= 9
        assert audit['undefined_references'] == audit['undefined_citations'] == audit['overfull_boxes'] == 0
        for name, expected_hash in audit['source_sha256'].items():
            assert hashlib.sha256((PAPER / name).read_bytes()).hexdigest() == expected_hash, (
                f'Compiled source changed: {name}; rebuild and refresh the compilation audit')
        paths = sorted([*paths, pdf, audit_path])
    write_anonymous_archive(output, paths, PAPER)
    with ZipFile(output) as archive:
        assert archive.testzip() is None
        assert "main.tex" in archive.namelist()
        assert not any("_zh" in name.lower() for name in archive.namelist())
        assert "appendix/contents.tex" in archive.namelist()
        assert "appendix/algorithm.tex" in archive.namelist()
        assert len([name for name in archive.namelist()
                    if name.startswith("figures/environments/") and name.endswith(".png")]) == 14
        assert "figures/final_success.pdf" not in archive.namelist()
        assert "figures/learning_efficiency.pdf" not in archive.namelist()
        assert "figures/paired_learning.pdf" not in archive.namelist()
        assert "figures/qrl_nqrl_comparison.pdf" not in archive.namelist()
        assert "figures/anqrl_revision/qrl_limitations.pdf" in archive.namelist()
        assert "figures/anqrl_revision/aqrl_overview.pdf" in archive.namelist()
        assert "figures/goal_fraction_success.pdf" in archive.namelist()
        assert "figures/task_success_grid.pdf" in archive.namelist()
        assert "figures/task_environments_grid.pdf" in archive.namelist()
        assert "figures/task_learning_curves.tex" not in archive.namelist()
        assert "figures/task_environments_1.pdf" not in archive.namelist()
        assert "figures/task_environments_2.pdf" not in archive.namelist()
        assert 'data/baseline_validation.csv' not in archive.namelist()
        assert 'tables/dynamics_mse_trajectory.tex' not in archive.namelist()
        assert 'tables/dynamics_mse_rescue.tex' not in archive.namelist()
        assert 'tables/execution_metrics.tex' not in archive.namelist()
        assert not any('first_arrival' in name or 'execution_metrics' in name
                       for name in archive.namelist())
        assert 'tables/goal_dimension_analysis.tex' in archive.namelist()
        assert 'tables/seed_test_success.tex' in archive.namelist()
        assert 'data/seed_test_success.csv' in archive.namelist()
        assert 'tables/search_all_tasks.tex' not in archive.namelist()
        assert 'data/goal_distance_steps_all_tasks.csv' not in archive.namelist()
        assert 'tables/search_seed_success.tex' in archive.namelist()
        assert 'data/goal_distance_steps_runs.csv' in archive.namelist()
        assert 'data/goal_distance_steps_provenance.json' in archive.namelist()
    print(f"Packaged {len(paths) + 1} files: {output} ({output.stat().st_size:,} bytes)")
    if args.sources_output:
        source_paths = set(paths)
        tool_names = (
            'draw_nqrl_network_figures.py', 'draw_nqrl_environment_gallery.py',
            'draw_nqrl_training_evidence.py', 'render_nqrl_environments.py',
            'nqrl_figure_style.py', 'nqrl_latex_figure.py', 'audit_nqrl_typography.py',
            'nqrl_table_labels.py', 'analyze_nqrl_paper_results.py',
            'audit_nqrl_network.py', 'package_nqrl_paper.py', 'qrl_env.sh',
            'prepare_nqrl_navigation_figure.py', 'write_nqrl_dynamics_tables.py',
            'analyze_dynamics_mse_evidence.py',
            'analyze_nqrl_execution_metrics.py',
            'analyze_nqrl_goal_dimensions.py', 'check_nqrl_paper.py',
            'analyze_nqrl_baseline_validation.py',
            'write_nqrl_seed_table.py', 'write_nqrl_search_ablation.py',
        )
        source_paths.update(ROOT / 'tools' / name for name in tool_names)
        source_paths.update((ROOT / 'quasimetric_rl').rglob('*.py'))
        source_paths.update((ROOT / 'online').glob('*.py'))
        source_paths.add(ROOT / 'tests/test_nqrl_first_arrival.py')
        source_paths.add(ROOT / 'tests/test_online_goal_envs.py')
        source_paths.add(ROOT / 'tests/test_paper_anonymization.py')
        source_paths.update((ROOT / 'requirements').glob('*.txt'))
        source_paths.update((PAPER / 'figures/environments').glob('*.json'))
        source_paths.add(PAPER / 'FIGURE_REPRODUCTION.md')
        source_paths.add(PAPER / 'FIGURE2_IMAGE_PROMPT.md')
        source_paths.add(PAPER / 'figures/navigation_goal_concept.png')
        source_paths.add(PAPER / 'data/dynamics_mse_runs.csv')
        source_paths.update((PAPER / 'data').glob('*'))
        source_paths.update((PAPER / 'archive').glob('*.tex'))
        source_paths.update((PAPER / 'archive').glob('*.md'))
        source_paths.add(ROOT / 'docs/GCRL_BASELINES.md')
        source_paths.add(PAPER / 'README.md')
        assert args.sources_output.resolve() != output.resolve()
        assert all(path.is_file() and path.resolve().is_relative_to(ROOT.resolve())
                   for path in source_paths)
        write_anonymous_archive(args.sources_output, sorted(source_paths), ROOT)
        with ZipFile(args.sources_output) as archive:
            assert archive.testzip() is None
            assert len(archive.namelist()) == len(source_paths) + 1
            assert 'tools/render_nqrl_environments.py' in archive.namelist()
            assert 'online/main.py' in archive.namelist()
            assert 'online/trainer.py' in archive.namelist()
            assert 'tests/test_nqrl_first_arrival.py' in archive.namelist()
            assert 'iclr2027/main.tex' in archive.namelist()
            assert not any('_zh' in name.lower() for name in archive.namelist())
        print(f"Packaged {len(source_paths) + 1} source files: {args.sources_output} "
              f"({args.sources_output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
