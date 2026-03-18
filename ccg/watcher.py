"""
File watcher: monitor repo for changes and run incremental ingest (content hash + Merkle).
Uses watchdog with debouncing so rapid edits trigger a single update.
"""

import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Set

logger = logging.getLogger(__name__)

# Extensions we care about (same as parser default)
DEFAULT_EXTENSIONS = {".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs", ".c", ".cpp", ".h", ".hpp"}
WATCH_IGNORED_DIRS = frozenset({
    ".git", ".svn", ".hg", "__pycache__", ".venv", "venv", "env",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "build", "dist",
    "node_modules", ".next", ".nuxt", "out", ".cache", "target",
    ".idea", ".vscode", ".cursor", "vendor",
})


def _should_ignore_event(src_path: str, root: Path, extensions: Optional[Set[str]] = None) -> bool:
    """True if we should ignore this path (won't trigger ingest)."""
    ext_set = extensions or DEFAULT_EXTENSIONS
    try:
        p = Path(src_path).resolve()
        if not p.exists():
            return False
        if p.is_file():
            if p.suffix.lower() not in ext_set:
                logger.debug("Ignored (extension not watched): %s (watched: %s)", src_path, sorted(ext_set))
                return True
            try:
                rel = p.relative_to(root)
            except ValueError:
                return True
            for part in rel.parts:
                if part in WATCH_IGNORED_DIRS:
                    logger.debug("Ignored (dir in watch list): %s", src_path)
                    return True
                if part.startswith(".") and part not in (".github", ".gitignore", ".env.example"):
                    logger.debug("Ignored (hidden dir): %s", src_path)
                    return True
            return False
        if p.is_dir():
            try:
                rel = p.relative_to(root)
            except ValueError:
                return True
            return rel.parts[0] in WATCH_IGNORED_DIRS if rel.parts else True
    except Exception:
        return True
    return False


def run_watcher(
    root_path: str | Path,
    config: Optional[Dict[str, Any]] = None,
    debounce_seconds: float = 2.0,
    extensions: Optional[Set[str]] = None,
) -> None:
    """
    Watch root_path for file changes; run incremental ingest after a quiet period.
    Blocks until keyboard interrupt (Ctrl+C).
    """
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

    root = Path(root_path).resolve()
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")

    lock = threading.Lock()
    pending = [False]
    last_trigger = [0.0]

    def schedule_ingest(src_path: str = "") -> None:
        with lock:
            pending[0] = True
            last_trigger[0] = time.monotonic()
        if src_path:
            logger.info("File change detected: %s (ingest in %.1fs)", src_path, debounce_seconds)

    def run_pending_ingest() -> None:
        from ccg.runner import ingest_codebase
        logger.info("Running incremental ingest after file change ...")
        try:
            result = ingest_codebase(root_path, config=config, incremental=True)
            if result.get("error"):
                logger.error("Ingest error: %s", result["error"])
            elif result.get("files_unchanged"):
                logger.info("No file changes applied (manifest up to date)")
            else:
                logger.info("Ingest result: %s", result)
        except Exception as e:
            logger.exception("Ingest failed: %s", e)

    class Handler(FileSystemEventHandler):
        def _check(self, event) -> None:
            if event.is_directory:
                return
            if _should_ignore_event(event.src_path, root, extensions):
                return
            schedule_ingest(event.src_path)

        def on_modified(self, event):
            self._check(event)

        def on_created(self, event):
            self._check(event)

        def on_deleted(self, event):
            if event.is_directory:
                return
            if _should_ignore_event(event.src_path, root, extensions):
                return
            schedule_ingest(event.src_path)

    observer = Observer()
    handler = Handler()
    observer.schedule(handler, str(root), recursive=True)
    observer.start()
    logger.info("Watching %s (debounce=%.1fs). Press Ctrl+C to stop.", root, debounce_seconds)

    try:
        while observer.is_alive():
            time.sleep(0.5)
            do_run = False
            with lock:
                if pending[0] and (time.monotonic() - last_trigger[0]) >= debounce_seconds:
                    pending[0] = False
                    do_run = True
            if do_run:
                run_pending_ingest()
    except KeyboardInterrupt:
        pass
    finally:
        observer.stop()
        observer.join()
        with lock:
            do_final = pending[0]
            pending[0] = False
        if do_final:
            run_pending_ingest()
        logger.info("Watcher stopped")
