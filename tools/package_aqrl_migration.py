#!/usr/bin/env python3
"""Archive paper runs and research sources for a private server migration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import lzma
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import time


REPO = Path(__file__).resolve().parents[1]
HOME_ROOT = REPO.parent
PREFIX = "AQRL_rebuttal_backup"
CACHE_DIRS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
              ".venv", "node_modules"}
WEIGHT_SUFFIXES = {".pth", ".pt", ".ckpt", ".safetensors"}
SOURCE_DIRS = (
    ".git", "aqrl_iclr2027_anonymous", "aqrl_iclr2027_submission",
    "base-qrl", "configs", "docs", "iclr2027", "offline", "online",
    "quasimetric_rl", "requirements", "tests", "third_party", "tools",
)
REQUIRED_RECORDS = ("test.log", "eval.log", "best_checkpoint.json",
                    "config.yaml", "git_summary.yaml")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def archive_name(path):
    return f"{PREFIX}/home/{path.relative_to(HOME_ROOT).as_posix()}"


def paper_dependencies(entry):
    """Follow the local TeX inputs and graphic assets used by one document."""
    pending = [entry]
    found = {entry}
    base = entry.parent
    pattern = re.compile(r"\\(input|include|includegraphics\*?|bibliography|bibliographystyle)"
                         r"\s*(?:\[[^\]]*\]\s*)?\{([^{}]+)\}")
    while pending:
        path = pending.pop()
        text = re.sub(r"(?<!\\)%[^\n]*", "", path.read_text())
        for command, value in pattern.findall(text):
            suffixes = ((".pdf", ".png", ".jpg", ".jpeg") if command.startswith("includegraphics")
                        else (".bib",) if command == "bibliography"
                        else (".bst",) if command == "bibliographystyle" else (".tex",))
            for value in value.split(","):
                target = Path(value.strip())
                candidates = [target] if target.suffix else [target.with_suffix(s) for s in suffixes]
                resolved = next((p.resolve() for directory in (base, path.parent)
                                 for name in candidates if (p := directory / name).is_file()), None)
                if resolved is None:
                    raise FileNotFoundError(f"Missing paper dependency in {path}: {value}")
                if resolved not in found:
                    found.add(resolved)
                    if resolved.suffix == ".tex":
                        pending.append(resolved)
    return found


def collect(*, compact=False):
    release = REPO / "aqrl_iclr2027_anonymous"
    main = read_csv(release / "code/experiments/main_source_run_ids.csv")
    ablation = read_csv(release / "results/goal_distance_steps_runs.csv")
    checks = json.loads((release / "results/baseline_reproduction_checks.json").read_text())
    groups = {}

    def include_run(run_id, study):
        if not run_id or Path(run_id).name != run_id:
            raise ValueError(f"Invalid run ID: {run_id!r}")
        groups.setdefault(run_id, set()).add(study)

    for row in main:
        include_run(row["archived_run_id"], "main")
    for row in ablation:
        include_run(row["task_id"], "search_ablation")
    for row in checks:
        run_id = row["source_run_id"]
        if row["algorithm"] == "CRL" and row["task"] == "PointSmall":
            # The release references this run by its historical queue number.
            run_id = "crl_quick_v1_CRL_CRLPointSmall_s1000"
        include_run(run_id, "baseline_check")
    if (len(main), len(ablation), len(checks), len(groups)) != (420, 120, 20, 530):
        raise ValueError("Paper run scope changed; review the migration selection first")

    entries = {}
    selected = []
    totals = {}

    def add(path, category):
        info = path.lstat()
        if not (stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)):
            raise ValueError(f"Unsupported file type: {path}")
        name = archive_name(path)
        if name not in entries:
            entries[name] = (path, category)
            totals[category] = totals.get(category, 0) + info.st_size

    def tree(root, category, *, optional=False, excluded=()):
        if not root.exists():
            if optional:
                return
            raise FileNotFoundError(root)
        for directory, dirs, files in os.walk(root, followlinks=False):
            parent = Path(directory)
            kept = []
            for name in sorted(dirs):
                path = parent / name
                relative = path.relative_to(root).as_posix()
                if name in CACHE_DIRS or relative in excluded:
                    continue
                if path.is_symlink():
                    add(path, category)
                else:
                    kept.append(name)
            dirs[:] = kept
            for name in sorted(files):
                path = parent / name
                if path.suffix in WEIGHT_SUFFIXES | {".pyc", ".pyo"}:
                    continue
                if category == "paper_run_records" and name.lower().startswith("replay"):
                    continue
                add(path, category)

    for directory in SOURCE_DIRS:
        if compact and directory == "iclr2027":
            continue
        category = "paper_sources" if directory == "iclr2027" else "research_code"
        tree(REPO / directory, category,
             excluded=("results",) if directory == "online" else ())
    if compact:
        paper = REPO / "iclr2027"
        dependencies = paper_dependencies(paper / "main.tex")
        for path in dependencies:
            add(path, "paper_sources")
        for path in paper.iterdir():
            if path.is_file() and path.suffix in {".md", ".sty", ".bst"}:
                add(path, "paper_sources")
        for directory in ("data", "reviews"):
            tree(paper / directory, "paper_sources")
        for graphic in sorted(p for p in dependencies if p.suffix == ".pdf"):
            for suffix in (".tex", ".svg", ".json"):
                source = graphic.with_suffix(suffix)
                if source.is_file():
                    for path in paper_dependencies(source) if suffix == ".tex" else (source,):
                        add(path, "paper_sources")
            labels = graphic.parent / "latex_sources" / graphic.stem
            if labels.is_dir():
                for path in labels.iterdir():
                    if path.is_file() and path.suffix in {".tex", ".json", ".pdf"}:
                        add(path, "paper_sources")
                        if path.suffix == ".tex":
                            for dependency in paper_dependencies(path):
                                add(dependency, "paper_sources")
        current_figures = paper / "figures/anqrl_revision"
        for name in ("draw_limitations.py", "draw_overview.py", "README.md",
                     "geometry.json", "qrl_limitations_geometry.json"):
            add(current_figures / name, "paper_sources")
        # These raster assets are also consumed by the editable figure generators.
        for directory in ("figures/panda_hybrid_v02", "figures/environments"):
            for path in (paper / directory).iterdir():
                if path.is_file() and path.suffix in {".png", ".json"} and not path.name.startswith("figure_"):
                    add(path, "paper_sources")
    for path in sorted(REPO.iterdir()):
        if path.is_symlink():
            add(path, "research_code")
        elif path.is_file() and path.suffix not in {".zip", ".gz", ".xz", ".pyc"}:
            add(path, "research_code")

    for run_id, studies in sorted(groups.items()):
        directory = HOME_ROOT / "qrl-assets/results/queue" / run_id
        if not directory.is_dir():
            directory = HOME_ROOT / run_id
        for filename in REQUIRED_RECORDS:
            if not (directory / filename).is_file():
                raise FileNotFoundError(f"Missing paper record: {directory / filename}")
        tree(directory, "paper_run_records")
        status = REPO / "runs/qrl_queue/status" / f"{run_id}.status"
        if status.is_file():
            add(status, "scheduler_records")
        selected.append({"run_id": run_id, "studies": sorted(studies),
                         "directory": archive_name(directory),
                         "original_directory": str(directory)})

    console_count = 0
    if not compact:
        for path in sorted((REPO / "logs/qrl_queue").iterdir()):
            if path.is_file() and any(path.name.startswith(run_id + "_")
                                      or path.name == run_id + ".log" for run_id in groups):
                add(path, "paper_console_logs")
                console_count += 1
    tree(HOME_ROOT / "gcrl-reference-repos", "reference_sources")
    for directory in ("mujoco", "micromamba/envs/python39", "micromamba/envs/graphics",
                      "micromamba/bin"):
        tree(HOME_ROOT / "qrl-assets" / directory, "simulator_resources")
    tree(HOME_ROOT / ".local/texlive", "latex_resources", optional=True)
    for directory in (".local/share/fonts", ".fonts", ".config/fontconfig"):
        tree(HOME_ROOT / directory, "latex_resources", optional=True)

    return entries, selected, {"compact": compact, "paper_runs": len(groups), "console_logs": console_count,
                               "files_and_links": len(entries), "bytes_by_category": totals,
                               "uncompressed_bytes": sum(totals.values())}


def capture(command, *, cwd=REPO, required=False, timeout=120, include_stderr=True):
    try:
        result = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=timeout,
                                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
                                     "PIP_DISABLE_PIP_VERSION_CHECK": "1"})
        output = result.stdout
        if result.stderr and include_stderr:
            output += "\n[stderr]\n" + result.stderr
        if result.returncode:
            if required:
                raise RuntimeError(f"Command failed: {command}: {output}")
            output += f"\n[exit status {result.returncode}]\n"
        return output
    except (OSError, subprocess.TimeoutExpired) as exc:
        if required:
            raise
        return f"Unavailable: {exc}\n"


def metadata(selected, summary):
    python = REPO / ".venv/bin/python"
    freeze = subprocess.run([str(python), "-m", "pip", "freeze", "--all",
                             "--disable-pip-version-check"], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    if freeze.returncode or "torch==" not in freeze.stdout:
        raise RuntimeError("Cannot export the current training environment: " + freeze.stderr)
    files = {
        "metadata/paper_runs.json": json.dumps(selected, indent=2) + "\n",
        "metadata/summary.json": json.dumps({**summary, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                               "source_repo": str(REPO)}, indent=2) + "\n",
        "metadata/training_pip_freeze.txt": freeze.stdout,
        "metadata/training_pip_list.json": capture([str(python), "-m", "pip", "list", "--format=json"],
                                                    required=True, include_stderr=False),
        "metadata/system_python_pip_freeze.txt": capture([sys.executable, "-m", "pip", "freeze", "--all"]),
        "metadata/git_status.txt": capture(["git", "status", "--porcelain=v1", "--untracked-files=all"], required=True),
        "metadata/git_head.txt": capture(["git", "rev-parse", "HEAD"], required=True),
        "metadata/git_worktree.patch": capture(["git", "diff", "--binary", "HEAD"], required=True),
        "metadata/nvidia-smi.txt": capture(["nvidia-smi"]),
        "metadata/system_packages.tsv": capture(["dpkg-query", "-W", "-f=${Package}\t${Version}\n"]),
        "metadata/platform.txt": platform.platform() + "\n" + capture(["uname", "-a"]),
        "metadata/fonts.txt": capture(["fc-list"]),
        "metadata/training_runtime.txt": capture([str(python), "-c", "import sys, torch; print(sys.version); print('torch:', torch.__version__); print('CUDA:', torch.version.cuda); print('cuDNN:', torch.backends.cudnn.version()); print(torch.__config__.show())"], required=True),
    }
    for name, path in (("os-release.txt", Path("/etc/os-release")),
                       ("pyvenv.cfg", REPO / ".venv/pyvenv.cfg")):
        if path.is_file():
            files["metadata/" + name] = path.read_text()
    mamba = HOME_ROOT / "qrl-assets/micromamba/bin/micromamba"
    for name in ("python39", "graphics"):
        files[f"metadata/conda_{name}_explicit.txt"] = capture([
            str(mamba), "list", "-p", str(HOME_ROOT / "qrl-assets/micromamba/envs" / name), "--explicit"])
    for path in sorted((HOME_ROOT / "gcrl-reference-repos").iterdir()):
        if (path / ".git").is_dir():
            files[f"metadata/reference_{path.name}.txt"] = (
                capture(["git", "rev-parse", "HEAD"], cwd=path) +
                capture(["git", "status", "--short"], cwd=path))
    files["RESTORE.txt"] = """AQRL private research backup

This is a personal migration archive, not the anonymous supplementary ZIP.
It contains paper sources, current code including uncommitted files and Git
history, 530 paper runs without weights/replay buffers,
reference sources, simulator resources, and environment records.

Verify the external SHA-256 before deleting the server originals. Extract on
Linux using: tar -xJf <archive-name>.tar.xz
The home/ folder preserves the project's sibling-directory layout, including
the five imported C-Learning Pusher runs. The original absolute paths are also
recorded in metadata/paper_runs.json. Update machine-specific paths in local
scheduler settings before running on another server.

Use home/quasimetric-rl/ as the research working directory. Recreate its .venv
from the saved setup scripts and metadata/training_pip_freeze.txt; the old
.venv is deliberately omitted. The saved Python/simulator installations are
reference copies and may require reinstalling or relocation on a new system.
metadata/ also records CUDA, cuDNN, operating-system packages, fonts and Git.
The anonymous reproduction setup is documented in
home/quasimetric-rl/aqrl_iclr2027_anonymous/code/README.md.
The paper entry point is home/quasimetric-rl/iclr2027/main.tex.

No model weights or replay buffers are included. Extra evaluation requiring a
trained model will require retraining. best_checkpoint.json is retained as
evaluation metadata, not as a model. No unreported run directories are included.
SSH private keys, caches and unrelated projects are not included.

metadata/manifest.json records each archived file's SHA-256 and each symbolic
link's target. The packaging script checks every member before declaring success.
Keep another copy of this archive if possible.
"""
    files["RESTORE.txt"] += (
        "\nCompact scope: console progress logs, manuscript history, old PDF/ZIP\n"
        "bundles, and unused figure previews are omitted. Current paper sources,\n"
        "active figures and their editable assets, evaluation logs, TensorBoard,\n"
        "and per-run code/configuration records are retained. The submitted PDF\n"
        "and supplementary ZIP are in home/quasimetric-rl/aqrl_iclr2027_submission/.\n"
        if summary.get("compact") else
        "\nFull scope: console logs and manuscript revision history are included.\n"
    )
    return {f"{PREFIX}/{name}": text.encode("utf-8") for name, text in files.items()}


class HashReader:
    def __init__(self, handle):
        self.handle = handle
        self.digest = hashlib.sha256()

    def read(self, size=-1):
        data = self.handle.read(size)
        self.digest.update(data)
        return data


def same_snapshot(before, after):
    return (before.st_size, before.st_mtime_ns, before.st_ino) == (
        after.st_size, after.st_mtime_ns, after.st_ino)


def add_bytes(archive, name, data):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = 0o600
    info.mtime = int(time.time())
    archive.addfile(info, io.BytesIO(data))
    return {"path": name, "type": "file", "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def verify(path):
    expected = None
    actual = []
    last_message = time.monotonic()
    with lzma.open(path, "rb") as compressed:
        with tarfile.open(fileobj=compressed, mode="r|") as archive:
            for info in archive:
                parts = PurePosixPath(info.name).parts
                if not parts or parts[0] != PREFIX or ".." in parts:
                    raise ValueError(f"Unexpected archive path: {info.name}")
                if info.name == f"{PREFIX}/metadata/manifest.json":
                    if expected is not None:
                        raise ValueError("Duplicate manifest")
                    expected = json.load(archive.extractfile(info))
                elif info.issym():
                    actual.append({"path": info.name, "type": "symlink", "target": info.linkname})
                elif info.isfile():
                    reader = HashReader(archive.extractfile(info))
                    while reader.read(1024 * 1024):
                        pass
                    actual.append({"path": info.name, "type": "file", "size": info.size,
                                   "sha256": reader.digest.hexdigest()})
                else:
                    raise ValueError(f"Unexpected archive member: {info.name}")
                if time.monotonic() - last_message > 30:
                    print(f"Verified {len(actual)} archive entries...", flush=True)
                    last_message = time.monotonic()
        while compressed.read(1024 * 1024):
            pass
    if expected is None or actual != expected or len({r['path'] for r in actual}) != len(actual):
        raise ValueError("Archive content does not match its manifest")
    print(f"Verified all {len(actual)} archive entries and the XZ stream.", flush=True)


def package(output, entries, extra):
    if not shutil.which("xz"):
        raise RuntimeError("xz is required")
    output = output.resolve()
    partial = output.with_name(output.name + ".partial")
    checksum = output.with_name(output.name + ".sha256")
    for path in (output, partial, checksum):
        if path.exists():
            raise FileExistsError(f"Will not overwrite {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = []
    proc = None
    try:
        with partial.open("xb") as destination:
            proc = subprocess.Popen(["xz", "-9e", "--threads=1", "--check=crc64", "-c"],
                                    stdin=subprocess.PIPE, stdout=destination)
            with tarfile.open(fileobj=proc.stdin, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                last_message = time.monotonic()
                for index, (name, (path, _category)) in enumerate(sorted(entries.items()), 1):
                    before = path.lstat()
                    info = archive.gettarinfo(str(path), arcname=name)
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    if info.issym():
                        archive.addfile(info)
                        manifest.append({"path": name, "type": "symlink", "target": info.linkname})
                    else:
                        # Store hard-linked source files independently for portability.
                        info.type = tarfile.REGTYPE
                        info.linkname = ""
                        info.size = before.st_size
                        with path.open("rb") as handle:
                            reader = HashReader(handle)
                            archive.addfile(info, reader)
                        manifest.append({"path": name, "type": "file", "size": info.size,
                                         "sha256": reader.digest.hexdigest()})
                    if not same_snapshot(before, path.lstat()):
                        raise RuntimeError(f"File changed while being archived: {path}")
                    if time.monotonic() - last_message > 30:
                        print(f"Packed {index}/{len(entries)} files; current: {path.name}", flush=True)
                        last_message = time.monotonic()
                for name, data in sorted(extra.items()):
                    manifest.append(add_bytes(archive, name, data))
                add_bytes(archive, f"{PREFIX}/metadata/manifest.json",
                          (json.dumps(manifest, indent=2) + "\n").encode())
            proc.stdin.close()
            if proc.wait() != 0:
                raise RuntimeError("xz compression failed")
            destination.flush()
            os.fsync(destination.fileno())
        print("Compression finished; verifying archive contents...", flush=True)
        verify(partial)
        digest = hashlib.sha256()
        with partial.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        partial.rename(output)
        with checksum.open("x") as handle:
            handle.write(f"{digest.hexdigest()}  {output.name}\n")
        print(f"SUCCESS: {output}\nSize: {output.stat().st_size / 1024**3:.3f} GiB\nSHA-256: {digest.hexdigest()}\nChecksum file: {checksum}", flush=True)
    except BaseException:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            proc.wait()
        print(f"Backup not completed. Any incomplete output remains at {partial}.", file=sys.stderr)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true", help="Validate selection without writing an archive")
    action.add_argument("--output", type=Path, help="Create a tar.xz archive and SHA-256 file")
    action.add_argument("--verify", type=Path, help="Verify an existing archive")
    parser.add_argument("--compact", action="store_true",
                        help="Omit console logs and keep only current paper sources/assets")
    args = parser.parse_args()
    os.umask(0o077)
    if args.verify:
        verify(args.verify)
        return
    entries, selected, summary = collect(compact=args.compact)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Uncompressed: {summary['uncompressed_bytes'] / 1024**3:.3f} GiB", flush=True)
    extra = metadata(selected, summary)
    print("Paper run records, source selection, and environment exports checked.", flush=True)
    if args.output:
        package(args.output, entries, extra)


if __name__ == "__main__":
    main()
