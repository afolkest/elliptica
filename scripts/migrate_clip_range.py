#!/usr/bin/env python3
"""One-off migration for clip_low/clip_high and legacy .flowcol files."""

from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable, Tuple


def _iter_project_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        if path.suffix not in {".elliptica", ".flowcol"}:
            continue
        yield path


def _load_metadata(zf: zipfile.ZipFile) -> dict:
    try:
        raw = zf.read("metadata.json")
    except KeyError as exc:
        raise RuntimeError("missing metadata.json") from exc
    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError("invalid metadata.json") from exc


def _update_display_settings(metadata: dict) -> Tuple[bool, str | None]:
    display = metadata.get("display_settings")
    if display is None:
        return False, "missing display_settings"

    if "clip_percent" not in display:
        return False, None

    clip_value = display.pop("clip_percent")
    display["clip_low_percent"] = clip_value
    display["clip_high_percent"] = clip_value
    return True, None


def _write_updated_zip(src_path: Path, dst_path: Path, metadata: dict) -> None:
    with tempfile.NamedTemporaryFile(delete=False, dir=dst_path.parent) as tmp:
        tmp_path = Path(tmp.name)

    try:
        with zipfile.ZipFile(src_path, "r") as src, zipfile.ZipFile(
            tmp_path, "w", zipfile.ZIP_DEFLATED
        ) as dst:
            for info in src.infolist():
                if info.filename == "metadata.json":
                    dst.writestr("metadata.json", json.dumps(metadata, indent=2))
                else:
                    dst.writestr(info, src.read(info.filename))
        tmp_path.replace(dst_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def migrate_file(path: Path, overwrite: bool, dry_run: bool) -> Tuple[str, str]:
    target_path = path.with_suffix(".elliptica") if path.suffix == ".flowcol" else path
    if target_path.exists() and target_path != path and not overwrite:
        return "skip", f"target exists: {target_path.name}"

    try:
        with zipfile.ZipFile(path, "r") as zf:
            metadata = _load_metadata(zf)
    except (zipfile.BadZipFile, RuntimeError) as exc:
        return "error", f"{path.name}: {exc}"

    updated, error = _update_display_settings(metadata)
    needs_rename = path.suffix == ".flowcol"
    if error:
        return "error", f"{path.name}: {error}"

    if not updated and not needs_rename:
        return "skip", f"{path.name}: already updated"

    if dry_run:
        action = "rename+update" if needs_rename and updated else "rename" if needs_rename else "update"
        return "dry-run", f"{path.name}: {action}"

    _write_updated_zip(path, target_path, metadata)
    if needs_rename and target_path != path:
        path.unlink()
    action = "renamed+updated" if needs_rename and updated else "renamed" if needs_rename else "updated"
    return "ok", f"{path.name}: {action}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Migrate clip_percent to clip_low/high and rename .flowcol -> .elliptica."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default="projects",
        help="Root directory containing project files (default: projects)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned changes without writing files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite .elliptica targets when migrating .flowcol files",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"Root does not exist: {root}")
        return 1

    results = {"ok": 0, "skip": 0, "error": 0, "dry-run": 0}
    for path in _iter_project_files(root):
        status, message = migrate_file(path, overwrite=args.overwrite, dry_run=args.dry_run)
        results[status] = results.get(status, 0) + 1
        print(f"[{status}] {message}")

    print(
        "Done:",
        f"ok={results['ok']}",
        f"skip={results['skip']}",
        f"dry-run={results['dry-run']}",
        f"error={results['error']}",
    )
    return 0 if results["error"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
