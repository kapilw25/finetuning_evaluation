"""
Advanced Logging System - Captures ALL terminal output to log files
Extracted from CITA PBT script for reuse across all training scripts

Usage:
    from logging_utils import setup_training_logger, restore_logging

    # At start of script
    log_file = setup_training_logger(run_name="SFT_Baseline", project_root=Path(__file__).parent.parent.parent)

    # ... your training code here ...

    # At end of script (in finally block)
    restore_logging(log_file)

Features:
- Tee class: Writes to both terminal AND log file simultaneously
- Captures stdout + stderr (all print statements, errors, warnings)
- Line buffering: Immediate writes (no buffering delays)
- Works like Unix 'tee' command
- Compatible with any training framework
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, TextIO


class Tee:
    """
    Tee class to write output to both terminal and log file
    Like Unix 'tee' command: captures everything to log file

    Usage:
        python comparative_study/01a_SFT_Baseline/Llama3_BF16.py

    Or for guaranteed unbuffered output:
        python -u comparative_study/01a_SFT_Baseline/Llama3_BF16.py
    """
    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()  # Ensure immediate display on terminal
        self.log_file.write(message)
        self.log_file.flush()  # Ensure immediate write to disk

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self):
        return self.terminal.isatty()

    def fileno(self):
        """Return file descriptor of terminal (required by faulthandler)"""
        return self.terminal.fileno()

    @property
    def encoding(self):
        """Return terminal encoding"""
        return getattr(self.terminal, 'encoding', 'utf-8')

    def close(self):
        """Close method (required by logging shutdown)"""
        # Don't close terminal, just close log file
        if hasattr(self.log_file, 'close'):
            self.log_file.close()


def setup_training_logger(
    run_name: str,
    project_root: Path,
    timestamp: Optional[str] = None
) -> tuple:
    """
    Setup advanced logging system (Tee) for training scripts

    Args:
        run_name: Name of training run (e.g., "SFT_Baseline", "DPO_Baseline", "CITA_Baseline")
        project_root: Path to project root directory
        timestamp: Optional custom timestamp (if None, auto-generates)

    Returns:
        Tuple of (log_file, log_filename_path, original_stdout, original_stderr)
        Store these for cleanup in finally block

    Usage:
        from logging_utils import setup_training_logger

        project_root = Path(__file__).parent.parent.parent
        log_file, log_path, orig_stdout, orig_stderr = setup_training_logger(
            run_name="SFT_Baseline",
            project_root=project_root
        )

        # ... training code ...

        # In finally block:
        restore_logging(log_file, orig_stdout, orig_stderr)
    """
    # Create logs directory
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)

    # Generate timestamped log filename
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = logs_dir / f"{run_name}_training_{timestamp}.log"

    # Open log file with line buffering (buffering=1) for real-time logging
    log_file = open(log_filename, 'w', buffering=1)

    # Save original stdout/stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # Redirect stdout and stderr to Tee
    sys.stdout = Tee(original_stdout, log_file)
    sys.stderr = Tee(original_stderr, log_file)

    print(f"📝 Logging initialized: {log_filename}")
    print(f"📝 All terminal output will be saved to this log file")
    print(f"📝 For guaranteed unbuffered output, run with: python -u <script.py>")
    print("="*80 + "\n")

    return log_file, log_filename, original_stdout, original_stderr


def restore_logging(log_file: TextIO, original_stdout, original_stderr):
    """
    Restore original stdout/stderr and close log file

    Args:
        log_file: Log file object from setup_training_logger()
        original_stdout: Original stdout from setup_training_logger()
        original_stderr: Original stderr from setup_training_logger()

    Usage:
        # In finally block at end of training script
        restore_logging(log_file, original_stdout, original_stderr)
        print(f"📝 Log file saved: {log_filename}")
    """
    # Restore original stdout/stderr
    sys.stdout = original_stdout
    sys.stderr = original_stderr

    # Close log file
    if log_file and not log_file.closed:
        log_file.close()
