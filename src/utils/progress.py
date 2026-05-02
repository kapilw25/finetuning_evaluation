"""tqdm progress-bar factory (CLAUDE.md §13: mandatory in every src/m*.py).

Usage:
    from src.utils.progress import make_pbar
    pbar = make_pbar(total=N, desc="aqi", unit="prompt")
    pbar.update(batch_size)  # call per batch
    pbar.close()  # at end
"""

from typing import Optional

from tqdm import tqdm


def make_pbar(
    total: int,
    desc: str,
    unit: str = "it",
    leave: bool = True,
    mininterval: float = 0.5,
    smoothing: float = 0.3,
    dynamic_ncols: bool = True,
    position: Optional[int] = None,
) -> tqdm:
    """Factory for a standardized tqdm bar across all CITA scripts.

    Defaults chosen for long GPU runs:
      - mininterval=0.5: refresh at most every 0.5s (avoids log flooding)
      - smoothing=0.3: rolling-average ETA (steadier than instantaneous)
      - dynamic_ncols=True: adapts to terminal width
    """
    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        leave=leave,
        mininterval=mininterval,
        smoothing=smoothing,
        dynamic_ncols=dynamic_ncols,
        position=position,
    )
