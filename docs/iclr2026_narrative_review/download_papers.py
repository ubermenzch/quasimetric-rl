#!/usr/bin/env python3
"""Download the ten official ICLR PDFs listed in the adjacent manifest."""

import argparse
import hashlib
import json
import socket
import sys
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path


BASE = Path(__file__).resolve().parent


def valid_pdf(path):
    if not path.is_file() or path.stat().st_size < 1024:
        return False
    with path.open("rb") as stream:
        if stream.read(5) != b"%PDF-":
            return False
        stream.seek(max(0, path.stat().st_size - 2048))
        return b"%%EOF" in stream.read()


def package(destination, papers):
    files = [BASE / name for name in (
        "README.md", "REPORT_ZH.md", "papers.json", "download_papers.py",
        "download_status.json",
    )]
    files.extend(BASE / "pdfs" / p["pdf_filename"] for p in papers
                 if valid_pdf(BASE / "pdfs" / p["pdf_filename"]))
    destination = destination.resolve()
    if destination in files:
        raise ValueError("ZIP destination must not overwrite a source file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            if path.is_file():
                archive.write(path, Path("iclr2026_narrative_review") / path.relative_to(BASE))
    print(f"ZIP: {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=30,
                        help="Network socket timeout in seconds (default: 30)")
    parser.add_argument("--zip", type=Path,
                        help="Package the report, manifest, status and available PDFs")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    papers = json.loads((BASE / "papers.json").read_text(encoding="utf-8"))["papers"]
    folder = BASE / "pdfs"
    folder.mkdir(exist_ok=True)
    results = []
    network_block = None
    for paper in papers:
        target = folder / paper["pdf_filename"]
        partial = target.with_suffix(".pdf.part")
        result = {"id": paper["id"], "url": paper["pdf_url"]}
        if valid_pdf(target):
            result["status"] = "already_present"
        elif network_block:
            result.update(status="not_attempted", error=network_block)
        else:
            print(f"Downloading {paper['id']}: {paper['title']}", flush=True)
            try:
                request = urllib.request.Request(
                    paper["pdf_url"], headers={"User-Agent": "ICLR-reading-pack/1.0"})
                with urllib.request.urlopen(request, timeout=args.timeout) as response:
                    with partial.open("wb") as stream:
                        while chunk := response.read(1024 * 1024):
                            stream.write(chunk)
                if not valid_pdf(partial):
                    raise ValueError("Response is not a complete PDF")
                partial.replace(target)
                result["status"] = "downloaded"
            except (OSError, urllib.error.URLError, ValueError) as error:
                result.update(status="failed", error=str(error))
                print(f"  FAILED: {error}", file=sys.stderr, flush=True)
                reason = getattr(error, "reason", error)
                if isinstance(reason, (socket.gaierror, PermissionError)):
                    network_block = f"Network unavailable: {reason}"
            finally:
                partial.unlink(missing_ok=True)
        if valid_pdf(target):
            result["bytes"] = target.stat().st_size
            result["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
        results.append(result)
    count = sum(r["status"] in ("downloaded", "already_present") for r in results)
    status = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "pdfs_available": count, "pdfs_expected": len(papers), "results": results,
    }
    (BASE / "download_status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Complete PDFs available: {count}/{len(papers)}")
    if args.zip:
        package(args.zip, papers)
    return 0 if count == len(papers) else 1


if __name__ == "__main__":
    sys.exit(main())
