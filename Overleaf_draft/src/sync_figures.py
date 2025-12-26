"""
Sync Evaluation Figures from outputs/ to Overleaf_draft/figures/

Copies ONLY whitelisted figures (defined in USED_FILES dict) that are
actually referenced via includegraphics in .tex files.

Usage:
    python Overleaf_draft/src/sync_figures.py          # Dry run - shows what would be copied
    python Overleaf_draft/src/sync_figures.py --sync   # Actually copy files

Output:
    [NEW]    - File doesn't exist in destination
    [UPDATE] - Source file is newer than destination (based on mtime)
    [SAME]   - Source file is same age or older than destination
    [SKIP]   - Directory not found or no whitelisted files
    [WARNING]- Expected file not found in source

To add/remove files: Edit the USED_FILES dict below.
"""

import shutil
import sys
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
OUTPUTS_EVAL = PROJECT_ROOT / "outputs" / "evaluation"
OVERLEAF_FIGURES = SCRIPT_DIR.parent / "figures" / "evaluation"

# =============================================================================
# WHITELIST: Only these files are used in .tex (from grep "includegraphics")
# =============================================================================
USED_FILES = {
    'AQI_Evaluation': [
        'aqi_comparison',
        'aqi_comparison_lollipop',
    ],
    'Conditional_Safety_Evaluation': [
        'conditional_safety_comparison',
        'conditional_safety_comparison_lollipop',
    ],
    'ISD_Evaluation_Embedding': [
        'isd_comparison',
        'isd_comparison_lollipop',
    ],
    'Length_Control_Evaluation': [
        'length_control_comparison',
        'length_control_comparison_lollipop',
    ],
    'TruthfulQA_Evaluation': [
        'truthfulqa_comparison',
        'truthfulqa_comparison_lollipop',
    ],
    'combined_plots': [
        'heatmap',
        'radar_area',
        # Note: teaser_single_prompt_idx287 is generated separately in outputs/evaluation/teaser_examples/
        # and already exists in Overleaf_draft/figures/evaluation/combined_plots/
    ],
}

# Source directories
SOURCE_DIRS = {
    'AQI_Evaluation': OUTPUTS_EVAL / 'AQI_Evaluation',
    'Conditional_Safety_Evaluation': OUTPUTS_EVAL / 'Conditional_Safety_Evaluation',
    'ISD_Evaluation_Embedding': OUTPUTS_EVAL / 'ISD_Evaluation_Embedding',
    'Length_Control_Evaluation': OUTPUTS_EVAL / 'Length_Control_Evaluation',
    'TruthfulQA_Evaluation': OUTPUTS_EVAL / 'TruthfulQA_Evaluation',
    'combined_plots': OUTPUTS_EVAL / 'combined_plots',
}

# File extensions to sync
FIGURE_EXTENSIONS = ['.pdf', '.png']


def get_whitelisted_files(src_dir: Path, whitelist: list) -> list:
    """Get only whitelisted figure files from a source directory."""
    files = []
    if src_dir.exists():
        for basename in whitelist:
            for ext in FIGURE_EXTENSIONS:
                filepath = src_dir / f"{basename}{ext}"
                if filepath.exists():
                    files.append(filepath)
    return files


def main():
    dry_run = '--sync' not in sys.argv

    print("=" * 70)
    print("SYNC EVALUATION FIGURES (Whitelisted Only)")
    print("=" * 70)
    print(f"Source: {OUTPUTS_EVAL}")
    print(f"Destination: {OVERLEAF_FIGURES}")
    print(f"Mode: {'DRY RUN (use --sync to actually copy)' if dry_run else 'SYNC MODE'}")
    print()

    # Count expected files
    total_basenames = sum(len(v) for v in USED_FILES.values())
    print(f"Whitelisted: {total_basenames} basenames × 2 extensions = {total_basenames * 2} files max")
    print()

    # Ensure destination directories exist
    OVERLEAF_FIGURES.mkdir(parents=True, exist_ok=True)
    (OVERLEAF_FIGURES / 'combined_plots').mkdir(parents=True, exist_ok=True)

    files_to_copy = []
    missing_files = []

    # Collect whitelisted files to copy
    for name, src_dir in SOURCE_DIRS.items():
        print(f"\n[{name}]")

        if not src_dir.exists():
            print(f"  [SKIP] Source directory not found: {src_dir}")
            continue

        # Determine destination
        if name == 'combined_plots':
            dest_dir = OVERLEAF_FIGURES / 'combined_plots'
        else:
            dest_dir = OVERLEAF_FIGURES

        # Get whitelisted files
        whitelist = USED_FILES.get(name, [])
        src_files = get_whitelisted_files(src_dir, whitelist)

        # Check for missing files
        for basename in whitelist:
            for ext in FIGURE_EXTENSIONS:
                expected = src_dir / f"{basename}{ext}"
                if not expected.exists():
                    missing_files.append(str(expected.relative_to(PROJECT_ROOT)))

        if not src_files:
            print(f"  [SKIP] No whitelisted files found")
            continue

        for src_file in src_files:
            dest_file = dest_dir / src_file.name
            files_to_copy.append((src_file, dest_file))

            # Check if file exists and compare modification time
            if dest_file.exists():
                src_mtime = src_file.stat().st_mtime
                dest_mtime = dest_file.stat().st_mtime
                if src_mtime > dest_mtime:
                    status = "[UPDATE]"
                else:
                    status = "[SAME]"
            else:
                status = "[NEW]"

            print(f"  {status} {src_file.name}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Files to sync: {len(files_to_copy)}")

    if missing_files:
        print(f"\n[WARNING] {len(missing_files)} expected files not found:")
        for mf in missing_files[:10]:  # Show first 10
            print(f"  - {mf}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")

    if not files_to_copy:
        print("\nNo files to sync!")
        return

    # Copy files
    if dry_run:
        print("\n[DRY RUN] No files copied. Run with --sync to actually copy.")
    else:
        print("\nCopying files...")
        copied_count = 0
        for src_file, dest_file in files_to_copy:
            try:
                shutil.copy2(src_file, dest_file)
                print(f"  [COPIED] {src_file.name} -> {dest_file.relative_to(OVERLEAF_FIGURES)}")
                copied_count += 1
            except Exception as e:
                print(f"  [ERROR] {src_file.name}: {e}")
        print(f"\nCopied {copied_count}/{len(files_to_copy)} files")


if __name__ == "__main__":
    main()
