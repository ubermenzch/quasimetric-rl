"""Read validated Git metadata even when Git emits valid output with exit 255."""
from pathlib import Path
import re
import subprocess


def git_output(args, *, cwd=None):
    args = list(args)
    status_query = args and args[0] == "status"
    if args not in (["rev-parse", "--show-toplevel"], ["rev-parse", "HEAD"],
                    ["status", "--short"], ["status", "--porcelain", "--untracked-files=no"]):
        raise ValueError(f"Unsupported metadata query: {args!r}")
    command = ["git", *args, *(["--branch"] if status_query else [])]
    result = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, check=False, timeout=30)
    output = result.stdout
    valid = result.returncode in (0, 255) and not result.stderr.strip()
    if status_query:
        lines = output.splitlines(keepends=True)
        valid = valid and bool(lines) and lines[0].startswith("## ")
        output = "".join(lines[1:])
        valid = valid and all(len(line) >= 3 and line[2] == " "
                              and all(ch in " MADRCUTXB?!" for ch in line[:2])
                              for line in output.splitlines())
    elif args[-1] == "HEAD":
        valid = valid and bool(re.fullmatch(r"[0-9a-fA-F]{40}(?:[0-9a-fA-F]{24})?", output.strip()))
    else:
        root = Path(output.strip())
        valid = valid and root.is_absolute() and (root / ".git").exists()
        if valid:
            try:
                Path(cwd or Path.cwd()).resolve().relative_to(root.resolve())
            except ValueError:
                valid = False
    if not valid:
        raise subprocess.CalledProcessError(result.returncode or 2, command,
                                            output=result.stdout,
                                            stderr=result.stderr or "Invalid Git metadata output")
    return output
