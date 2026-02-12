#!/usr/bin/env python3
"""Watch ~/Documents/ExportCal for file changes and auto-sync to the calendar.

When a CSV/Excel file is created or modified in the watched folder:
  1. Matches it to an existing source in config.json (by filename).
  2. Copies the file into the repo's data/ directory, overwriting the old version.
  3. Commits and pushes to GitHub so Streamlit Cloud redeploys automatically.

Usage:
    python sync_calendar.py          # watch ~/Documents/ExportCal (default)
    python sync_calendar.py /path    # watch a custom folder

Press Ctrl+C to stop.
"""

import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_WATCH = Path.home() / "Documents" / "ExportCal"
APP_DIR = Path(__file__).parent.resolve()
CONFIG_PATH = APP_DIR / "config.json"
DATA_DIR = APP_DIR / "data"
SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".tsv"}
DEBOUNCE_SECONDS = 2  # wait this long after last change before acting


# ---------------------------------------------------------------------------
# Helpers (mirrored from app.py)
# ---------------------------------------------------------------------------

def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    return {"sheets": [], "watch_folders": []}


def match_file_to_source(filename: str, config: dict) -> int | None:
    """Return the config['sheets'] index that matches *filename*, or None."""
    stem = Path(filename).stem
    for idx, s in enumerate(config.get("sheets", [])):
        if s.get("source_type") != "upload":
            continue
        stored_name = Path(s.get("file_path", "")).name
        stored_stem = Path(stored_name).stem
        if filename == stored_name:
            return idx
        base = re.sub(r"_\d+$", "", stored_stem)
        if stem == base or stem == stored_stem:
            return idx
    return None


def git_push(message: str):
    """Stage data/ changes, commit, and push."""
    try:
        subprocess.run(["git", "add", "data/"], cwd=APP_DIR, check=True)
        # Check if there's anything to commit
        result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=APP_DIR,
        )
        if result.returncode == 0:
            log("  No changes to commit (file identical to stored version).")
            return
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=APP_DIR, check=True,
        )
        subprocess.run(["git", "push"], cwd=APP_DIR, check=True)
        log("  Pushed to GitHub.")
    except subprocess.CalledProcessError as e:
        log(f"  Git error: {e}")


def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# File sync logic
# ---------------------------------------------------------------------------

def sync_file(filepath: Path):
    """Copy a changed file into data/ and push if it matches a source."""
    if filepath.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return
    if not filepath.is_file():
        return

    config = load_config()
    idx = match_file_to_source(filepath.name, config)

    if idx is None:
        log(f"  Skipped {filepath.name} (no matching source in config.json)")
        return

    source_name = config["sheets"][idx].get("name", f"Source {idx + 1}")
    dest = APP_DIR / config["sheets"][idx]["file_path"]

    # Copy the file
    shutil.copy2(filepath, dest)
    log(f"  Copied {filepath.name} -> {dest.relative_to(APP_DIR)}  ({source_name})")

    # Commit and push
    git_push(f"auto-sync: updated {filepath.name}")


# ---------------------------------------------------------------------------
# Watchdog handler with debouncing
# ---------------------------------------------------------------------------

class ExportCalHandler(FileSystemEventHandler):
    def __init__(self):
        super().__init__()
        self._pending: dict[str, float] = {}  # filepath -> timestamp of last event

    def _schedule(self, path: str):
        if Path(path).suffix.lower() in SUPPORTED_EXTENSIONS:
            self._pending[path] = time.time()

    def on_created(self, event):
        if not event.is_directory:
            self._schedule(event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            self._schedule(event.src_path)

    def on_moved(self, event):
        if not event.is_directory:
            self._schedule(event.dest_path)

    def process_pending(self):
        """Process any files that have been stable for DEBOUNCE_SECONDS."""
        now = time.time()
        ready = [p for p, t in self._pending.items() if now - t >= DEBOUNCE_SECONDS]
        for path in ready:
            del self._pending[path]
            fp = Path(path)
            if fp.exists():
                log(f"Detected change: {fp.name}")
                sync_file(fp)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    watch_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_WATCH

    if not watch_dir.is_dir():
        print(f"Creating watch folder: {watch_dir}")
        watch_dir.mkdir(parents=True, exist_ok=True)

    log(f"Watching: {watch_dir}")
    log(f"Repo:     {APP_DIR}")
    log("")

    # Show which filenames will be matched
    config = load_config()
    log("Configured sources (drop files with these names to sync):")
    for s in config.get("sheets", []):
        if s.get("source_type") == "upload":
            fname = Path(s.get("file_path", "")).name
            log(f"  - {fname}  ({s.get('name', '?')})")
    log("")
    log("Waiting for changes... (Ctrl+C to stop)")
    log("")

    handler = ExportCalHandler()
    observer = Observer()
    observer.schedule(handler, str(watch_dir), recursive=False)
    observer.start()

    try:
        while True:
            handler.process_pending()
            time.sleep(0.5)
    except KeyboardInterrupt:
        log("Stopping watcher.")
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
