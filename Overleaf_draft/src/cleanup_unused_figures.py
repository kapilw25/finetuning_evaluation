"""
Cleanup Unused Figures in Overleaf_draft/figures/

This script:
1. Scans all .tex files to find referenced PDF/PNG files
2. Keeps referenced files (both .pdf and .png versions)
3. DELETES unused files to reduce clutter

Usage:
    python Overleaf_draft/src/cleanup_unused_figures.py          # Dry run (shows what would be deleted)
    python Overleaf_draft/src/cleanup_unused_figures.py --delete # Actually delete files

"""

import re
import sys
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
OVERLEAF_DIR = SCRIPT_DIR.parent
FIGURES_DIR = OVERLEAF_DIR / "figures"

# File extensions to track
FIGURE_EXTENSIONS = {'.pdf', '.png', '.jpg', '.jpeg', '.csv'}


def find_tex_files() -> list:
    """Find all .tex files in Overleaf_draft/"""
    return list(OVERLEAF_DIR.glob("*.tex"))


def extract_figure_references(tex_file: Path) -> set:
    """Extract all figure paths referenced in a .tex file."""
    references = set()

    content = tex_file.read_text(encoding='utf-8', errors='ignore')

    # Pattern to match \includegraphics[...]{path} or \includegraphics{path}
    # Also matches paths with or without extension
    patterns = [
        r'\\includegraphics\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}',
        r'\\input\s*\{([^}]+)\}',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, content)
        for match in matches:
            # Normalize path
            path = match.strip()
            # Remove leading figures/ if present (we'll add it back)
            if path.startswith('figures/'):
                path = path[8:]
            references.add(path)

    return references


def get_all_figure_files() -> dict:
    """Get all files in figures/ directory, organized by stem (without extension)."""
    files = {}

    for file_path in FIGURES_DIR.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in FIGURE_EXTENSIONS:
            # Get relative path from figures/
            rel_path = file_path.relative_to(FIGURES_DIR)
            # Get stem (path without extension)
            stem = str(rel_path.with_suffix(''))

            if stem not in files:
                files[stem] = []
            files[stem].append(file_path)

    return files


def main():
    dry_run = '--delete' not in sys.argv

    print("=" * 70)
    print("CLEANUP UNUSED FIGURES")
    print("=" * 70)
    print(f"Overleaf directory: {OVERLEAF_DIR}")
    print(f"Figures directory: {FIGURES_DIR}")
    print(f"Mode: {'DRY RUN (use --delete to actually delete)' if dry_run else 'DELETE MODE'}")
    print()

    # Step 1: Find all .tex files
    tex_files = find_tex_files()
    print(f"Found {len(tex_files)} .tex files")

    # Step 2: Extract all figure references
    all_references = set()
    for tex_file in tex_files:
        refs = extract_figure_references(tex_file)
        all_references.update(refs)
        if refs:
            print(f"  {tex_file.name}: {len(refs)} references")

    # Normalize references (remove extensions to match stems)
    normalized_refs = set()
    for ref in all_references:
        # Remove extension if present
        ref_path = Path(ref)
        stem = str(ref_path.with_suffix(''))
        normalized_refs.add(stem)

    print(f"\nTotal unique figure stems referenced: {len(normalized_refs)}")

    # Step 3: Get all figure files
    all_files = get_all_figure_files()
    print(f"Total figure stems in directory: {len(all_files)}")

    # Step 4: Categorize files
    used_stems = set()
    unused_stems = set()

    for stem in all_files:
        if stem in normalized_refs:
            used_stems.add(stem)
        else:
            unused_stems.add(stem)

    # Step 5: Report
    print("\n" + "=" * 70)
    print("USED FILES (will be KEPT)")
    print("=" * 70)
    for stem in sorted(used_stems):
        files = all_files[stem]
        print(f"  {stem}")
        for f in files:
            print(f"    - {f.name}")

    print("\n" + "=" * 70)
    print("UNUSED FILES (will be DELETED)" if not dry_run else "UNUSED FILES (would be deleted)")
    print("=" * 70)

    files_to_delete = []
    for stem in sorted(unused_stems):
        files = all_files[stem]
        print(f"  {stem}")
        for f in files:
            print(f"    - {f.name}")
            files_to_delete.append(f)

    # Step 6: Delete (if not dry run)
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Used figure stems: {len(used_stems)}")
    print(f"Unused figure stems: {len(unused_stems)}")
    print(f"Files to delete: {len(files_to_delete)}")

    if files_to_delete:
        if dry_run:
            print("\n[DRY RUN] No files deleted. Run with --delete to actually delete.")
        else:
            print("\nDeleting files...")
            deleted_count = 0
            for f in files_to_delete:
                try:
                    f.unlink()
                    print(f"  [DELETED] {f.relative_to(FIGURES_DIR)}")
                    deleted_count += 1
                except Exception as e:
                    print(f"  [ERROR] {f}: {e}")
            print(f"\nDeleted {deleted_count}/{len(files_to_delete)} files")
    else:
        print("\nNo unused files found!")

    # Step 7: Check for empty directories
    print("\n" + "=" * 70)
    print("CHECKING EMPTY DIRECTORIES")
    print("=" * 70)
    for dir_path in sorted(FIGURES_DIR.rglob("*")):
        if dir_path.is_dir():
            contents = list(dir_path.iterdir())
            if not contents:
                if dry_run:
                    print(f"  [EMPTY] {dir_path.relative_to(FIGURES_DIR)} (would be removed)")
                else:
                    dir_path.rmdir()
                    print(f"  [REMOVED] {dir_path.relative_to(FIGURES_DIR)}")


if __name__ == "__main__":
    main()
