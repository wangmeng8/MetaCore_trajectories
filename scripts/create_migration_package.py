"""Create a clean cluster migration archive.

The package intentionally excludes local virtual environments, API-key files,
old run outputs, and cache directories even when they are nested inside cloned
benchmark repositories.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import tarfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ITEMS = [
    "patches",
    "docs",
    "scripts",
    "tests",
    "README.md",
    "pyproject.toml",
    ".env.example",
    ".gitignore",
    "MIGRATION_CLUSTER.md",
    "requirements-freeze-main.txt",
    "requirements-freeze-hle.txt",
    "data",
    ".external",
]

EXCLUDED_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    ".env",
    ".conda",
    "conda-env",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
    "._____temp",
    "outputs",
    "jobs",
    "logs",
    "dist",
    "node_modules",
}

EXCLUDED_FILE_NAMES = {
    ".env",
    ".DS_Store",
}

EXCLUDED_FILE_PATTERNS = {
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "*.log",
}


@dataclass
class PackageStats:
    included_files: int = 0
    included_bytes: int = 0
    skipped_dirs: int = 0
    skipped_files: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a clean MetaCoreBench migration tarball.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Archive path. Defaults to dist/MetaCoreBench_cluster_migration_<timestamp>_clean.tar.gz.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Project root to package. Defaults to the repository root.",
    )
    parser.add_argument("--dry-run", action="store_true", help="List what would be packaged without writing.")
    parser.add_argument("--force", action="store_true", help="Overwrite --output if it already exists.")
    return parser.parse_args()


def should_skip_file(path: Path) -> bool:
    if path.name in EXCLUDED_FILE_NAMES:
        return True
    return any(fnmatch.fnmatch(path.name, pattern) for pattern in EXCLUDED_FILE_PATTERNS)


def iter_package_files(root: Path, stats: PackageStats):
    for item in DEFAULT_ITEMS:
        start = root / item
        if not start.exists():
            continue

        if start.is_file():
            if should_skip_file(start):
                stats.skipped_files += 1
                continue
            yield start
            continue

        for current_root, dir_names, file_names in os.walk(start, topdown=True):
            current = Path(current_root)
            kept_dirs = []
            for dirname in dir_names:
                if dirname in EXCLUDED_DIR_NAMES:
                    stats.skipped_dirs += 1
                else:
                    kept_dirs.append(dirname)
            dir_names[:] = kept_dirs

            for filename in file_names:
                path = current / filename
                if should_skip_file(path):
                    stats.skipped_files += 1
                    continue
                yield path


def add_file(tar: tarfile.TarFile, path: Path, root: Path, stats: PackageStats) -> None:
    arcname = path.relative_to(root).as_posix()
    tar.add(path, arcname=arcname, recursive=False)
    stats.included_files += 1
    try:
        stats.included_bytes += path.stat().st_size
    except OSError:
        pass


def default_output(root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return root / "dist" / f"MetaCoreBench_cluster_migration_{timestamp}_clean.tar.gz"


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    output = (args.output or default_output(root)).resolve()
    stats = PackageStats()
    files = list(iter_package_files(root, stats))

    if args.dry_run:
        print(f"Project root: {root}")
        print(f"Archive path: {output}")
        print(f"Files to include: {len(files)}")
        print(f"Skipped directories: {stats.skipped_dirs}")
        print(f"Skipped files: {stats.skipped_files}")
        for path in files[:200]:
            print(path.relative_to(root).as_posix())
        if len(files) > 200:
            print(f"... {len(files) - 200} more files")
        return 0

    if output.exists() and not args.force:
        raise SystemExit(f"ERROR: output already exists: {output}. Use --force or choose another --output.")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, mode="w:gz") as tar:
        for path in files:
            add_file(tar, path, root, stats)

    print(f"Archive: {output}")
    print(f"Included files: {stats.included_files}")
    print(f"Included bytes before compression: {stats.included_bytes}")
    print(f"Skipped directories: {stats.skipped_dirs}")
    print(f"Skipped files: {stats.skipped_files}")
    print(f"Compressed size: {output.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
