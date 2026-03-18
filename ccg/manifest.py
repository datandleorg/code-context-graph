"""
File manifest with content hashing for incremental ingest.
Merkle-style root hash = hash of sorted (path + file_hash) to detect any change.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "manifest.json"


def file_content_hash(path: Path) -> str:
    """SHA-256 hash of file content (normalized line endings)."""
    content = path.read_text(encoding="utf-8", errors="replace")
    normalized = content.strip().replace("\r\n", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def compute_root_hash(files_hashes: Dict[str, str]) -> str:
    """Merkle-style root: hash of sorted path:hash pairs so any file change changes root."""
    if not files_hashes:
        return hashlib.sha256(b"").hexdigest()
    parts = "".join(f"{p}\0{h}" for p, h in sorted(files_hashes.items()))
    return hashlib.sha256(parts.encode("utf-8")).hexdigest()


def load_manifest(index_dir: Path) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    """
    Load manifest from index_dir/manifest.json.
    Returns (files_hashes dict path -> hash, root_hash) or (None, None) if missing.
    """
    path = index_dir / MANIFEST_FILENAME
    if not path.exists():
        return None, None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        files = data.get("files") or {}
        root = data.get("root_hash") or compute_root_hash(files)
        return files, root
    except Exception as e:
        logger.warning("Failed to load manifest %s: %s", path, e)
        return None, None


def save_manifest(index_dir: Path, files_hashes: Dict[str, str]) -> str:
    """Write manifest.json; return root_hash."""
    index_dir.mkdir(parents=True, exist_ok=True)
    root_hash = compute_root_hash(files_hashes)
    path = index_dir / MANIFEST_FILENAME
    path.write_text(
        json.dumps({"root_hash": root_hash, "files": files_hashes}, indent=0),
        encoding="utf-8",
    )
    logger.info("Saved manifest: %d files, root_hash=%s", len(files_hashes), root_hash[:16])
    return root_hash


def compute_file_hashes(
    root: Path,
    file_paths: List[Path],
) -> Dict[str, str]:
    """Compute path -> content_hash for each file (path as relative posix string)."""
    root_resolved = root.resolve()
    out: Dict[str, str] = {}
    for p in file_paths:
        try:
            rel = p.resolve().relative_to(root_resolved).as_posix()
            out[rel] = file_content_hash(p)
        except (ValueError, OSError) as e:
            logger.debug("Skip hash %s: %s", p, e)
    return out


def diff_manifest(
    previous: Optional[Dict[str, str]],
    current: Dict[str, str],
) -> Tuple[List[str], List[str], List[str]]:
    """
    Compare previous vs current (path -> hash).
    Returns (new_paths, changed_paths, deleted_paths) as relative posix strings.
    """
    prev_set = set(previous or {})
    curr_set = set(current.keys())
    new = list(curr_set - prev_set)
    deleted = list(prev_set - curr_set)
    changed = [p for p in curr_set & prev_set if previous[p] != current[p]]
    return new, changed, deleted
