#!/usr/bin/env python3
"""
Compare CITA_Instruct vs CITA_NoInstruct Training Progress

Parses training logs and generates comparison plots showing:
- eval/rewards/margins over training
- eval/loss over training
- Side-by-side comparison at matching checkpoints

Usage:
    python unit_test/compare_cita_training.py \
        logs/CITA_Instruct_training_20251116_222112.log \
        logs/CITA_NoInstruct_training_20251116_015222.log

Output:
    - Plot: logs_training/iter12/plots/Instruct_vs_NoInstruct_progress.png
    - Summary statistics printed to console
"""

import re
import sys
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def parse_training_log(log_path):
    """
    Parse CITA training log to extract progress and eval metrics

    Args:
        log_path: Path to training log file

    Returns:
        dict with keys:
            - current_step: Last training step
            - current_pct: Progress percentage
            - total_steps: Total training steps
            - evals: List of eval checkpoints with {epoch, loss, margin, accuracy}
            - last_train_loss: Last training loss
            - last_grad_norm: Last gradient norm
    """
    with open(log_path, 'r') as f:
        content = f.read()

    # Find all training progress lines (e.g., "60%|████| 810/1354")
    progress_pattern = r'(\d+)%\|.*?\|\s*(\d+)/(\d+)'
    progress_matches = re.findall(progress_pattern, content)

    if progress_matches:
        last_pct, last_step, total = progress_matches[-1]
        current_step = int(last_step)
        current_pct = int(last_pct)
        total_steps = int(total)
    else:
        current_step = 0
        current_pct = 0
        total_steps = 1354  # Default for CITA

    # Find all eval checkpoints
    # Pattern: {'eval_loss': 0.4247, ..., 'eval_rewards/margins': 5.7229, ..., 'epoch': 0.2}
    eval_pattern = r"'eval_loss':\s*([\d.]+).*?'eval_rewards/accuracies':\s*([\d.]+).*?'eval_rewards/margins':\s*([\d.]+).*?'epoch':\s*([\d.]+)"
    eval_matches = re.findall(eval_pattern, content)

    evals = []
    for loss, accuracy, margin, epoch in eval_matches:
        evals.append({
            'epoch': float(epoch),
            'loss': float(loss),
            'margin': float(margin),
            'accuracy': float(accuracy)
        })

    # Sort by epoch
    evals.sort(key=lambda x: x['epoch'])

    # Find training metrics (loss, grad_norm) from last few steps
    train_loss_pattern = r"'loss':\s*([\d.]+),\s*'grad_norm':\s*([\d.]+)"
    train_matches = re.findall(train_loss_pattern, content)

    last_train_loss = None
    last_grad_norm = None
    if train_matches:
        last_train_loss, last_grad_norm = train_matches[-1]
        last_train_loss = float(last_train_loss)
        last_grad_norm = float(last_grad_norm)

    return {
        'current_step': current_step,
        'current_pct': current_pct,
        'total_steps': total_steps,
        'evals': evals,
        'last_train_loss': last_train_loss,
        'last_grad_norm': last_grad_norm
    }


def print_summary(instruct_data, noinstruct_data):
    """Print detailed comparison summary"""
    print("=" * 80)
    print("TRAINING COMPARISON: CITA_Instruct vs CITA_NoInstruct")
    print("=" * 80)

    # Instruct progress
    print(f"\n📊 CITA_INSTRUCT:")
    print(f"  Progress: {instruct_data['current_step']}/{instruct_data['total_steps']} ({instruct_data['current_pct']}%)")
    if instruct_data['last_train_loss']:
        print(f"  Last train loss: {instruct_data['last_train_loss']:.4f}")
        print(f"  Last grad_norm: {instruct_data['last_grad_norm']:.4f}")
    print(f"  Eval checkpoints: {len(instruct_data['evals'])}")
    if instruct_data['evals']:
        for eval in instruct_data['evals']:
            pct = int(eval['epoch'] * 100)
            print(f"    {pct:3d}%: loss={eval['loss']:.4f}, margin={eval['margin']:.4f}, accuracy={eval['accuracy']:.4f}")

    # NoInstruct progress
    print(f"\n📊 CITA_NOINSTRUCT:")
    status = "COMPLETE" if noinstruct_data['current_pct'] >= 99 else f"{noinstruct_data['current_pct']}%"
    print(f"  Progress: {noinstruct_data['current_step']}/{noinstruct_data['total_steps']} ({status})")
    print(f"  Eval checkpoints: {len(noinstruct_data['evals'])}")
    if noinstruct_data['evals']:
        for eval in noinstruct_data['evals']:
            pct = int(eval['epoch'] * 100)
            print(f"    {pct:3d}%: loss={eval['loss']:.4f}, margin={eval['margin']:.4f}, accuracy={eval['accuracy']:.4f}")

    # Comparison at matching checkpoints
    if instruct_data['evals'] and noinstruct_data['evals']:
        print("\n" + "=" * 80)
        print("COMPARISON AT MATCHING CHECKPOINTS")
        print("=" * 80)
        print(f"{'Epoch':<10} {'Instruct Margin':<20} {'NoInstruct Margin':<20} {'Difference':<15}")
        print("-" * 80)

        for inst_eval in instruct_data['evals']:
            # Find matching NoInstruct eval (within 1% tolerance)
            matching = [e for e in noinstruct_data['evals'] if abs(e['epoch'] - inst_eval['epoch']) < 0.01]
            if matching:
                no_eval = matching[0]
                diff = inst_eval['margin'] - no_eval['margin']
                pct = int(inst_eval['epoch'] * 100)
                diff_pct = (diff / no_eval['margin'] * 100) if no_eval['margin'] > 0 else 0
                print(f"{pct}%{'':<7} {inst_eval['margin']:<20.4f} {no_eval['margin']:<20.4f} {diff:+.4f} ({diff_pct:+.1f}%)")

    # Projection
    if instruct_data['evals'] and noinstruct_data['evals'] and len(noinstruct_data['evals']) >= 2:
        print("\n" + "=" * 80)
        print("TRAJECTORY ANALYSIS & PROJECTION")
        print("=" * 80)

        # NoInstruct growth
        print("\n📈 CITA_NoInstruct trajectory:")
        no_margins = [e['margin'] for e in noinstruct_data['evals']]
        for i, eval in enumerate(noinstruct_data['evals']):
            pct = int(eval['epoch'] * 100)
            growth = f"(+{no_margins[i] - no_margins[i-1]:.2f})" if i > 0 else ""
            print(f"  {pct:3d}%: margin={eval['margin']:.2f} {growth}")

        if len(noinstruct_data['evals']) >= 2:
            growth_20_40 = no_margins[1] - no_margins[0]
            final_margin = no_margins[-1]
            print(f"\n  Peak growth (20% → 40%): +{growth_20_40:.2f}")
            print(f"  Final margin: {final_margin:.2f}")

        # Instruct projection
        if instruct_data['evals']:
            print("\n📊 CITA_Instruct projection:")
            inst_current = instruct_data['evals'][-1]
            print(f"  {int(inst_current['epoch'] * 100)}%: margin={inst_current['margin']:.2f} (CURRENT)")

            if len(noinstruct_data['evals']) >= 2 and inst_current['epoch'] < 0.3:
                # Project based on NoInstruct growth pattern
                growth_20_40 = no_margins[1] - no_margins[0]
                projected_40 = inst_current['margin'] + growth_20_40
                projected_final = projected_40 + (final_margin - no_margins[1])

                print(f"  Instruct starts {inst_current['margin'] - no_margins[0]:+.2f} higher than NoInstruct @ 20%")
                print(f"  If similar growth pattern (+{growth_20_40:.2f} at 40%):")
                print(f"    40%: margin ~{projected_40:.2f}")
                print(f"    100%: margin ~{projected_final:.2f} (PROJECTED)")

    print("\n" + "=" * 80)


def create_comparison_plot(instruct_data, noinstruct_data, output_path):
    """
    Create comparison plot showing margins and losses over training

    Args:
        instruct_data: Parsed Instruct training data
        noinstruct_data: Parsed NoInstruct training data
        output_path: Path to save plot
    """
    # Extract data
    inst_epochs = [e['epoch'] * 100 for e in instruct_data['evals']]
    inst_margins = [e['margin'] for e in instruct_data['evals']]
    inst_losses = [e['loss'] for e in instruct_data['evals']]

    no_epochs = [e['epoch'] * 100 for e in noinstruct_data['evals']]
    no_margins = [e['margin'] for e in noinstruct_data['evals']]
    no_losses = [e['loss'] for e in noinstruct_data['evals']]

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Margins
    if no_epochs and no_margins:
        ax1.plot(no_epochs, no_margins, 'o-', label='CITA_NoInstruct (COMPLETE)',
                color='green', linewidth=2, markersize=8)
        final_margin = no_margins[-1]
        ax1.axhline(y=final_margin, color='green', linestyle='--', alpha=0.3,
                   label=f'NoInstruct final ({final_margin:.2f})')

    if inst_epochs and inst_margins:
        ax1.plot(inst_epochs, inst_margins, 'o', label='CITA_Instruct (IN PROGRESS)',
                color='orange', markersize=12)

        # Annotate difference at first checkpoint
        if no_epochs and no_margins and inst_epochs[0] == no_epochs[0]:
            diff = inst_margins[0] - no_margins[0]
            if diff > 0:
                ax1.annotate(f'+{diff:.2f}\n(Instruct\nahead!)',
                           xy=(inst_epochs[0], inst_margins[0]),
                           xytext=(inst_epochs[0] + 5, inst_margins[0] - 1),
                           fontsize=10, color='red', fontweight='bold',
                           arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    ax1.set_xlabel('Training Progress (%)', fontsize=12)
    ax1.set_ylabel('eval/rewards/margins', fontsize=12)
    ax1.set_title('CITA Training Comparison: Margins', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(10, 110)

    # Plot 2: Loss
    if no_epochs and no_losses:
        ax2.plot(no_epochs, no_losses, 'o-', label='CITA_NoInstruct (COMPLETE)',
                color='green', linewidth=2, markersize=8)
        final_loss = no_losses[-1]
        ax2.axhline(y=final_loss, color='green', linestyle='--', alpha=0.3,
                   label=f'NoInstruct final ({final_loss:.2f})')

    if inst_epochs and inst_losses:
        ax2.plot(inst_epochs, inst_losses, 'o', label='CITA_Instruct (IN PROGRESS)',
                color='orange', markersize=12)

    ax2.set_xlabel('Training Progress (%)', fontsize=12)
    ax2.set_ylabel('eval/loss', fontsize=12)
    ax2.set_title('CITA Training Comparison: Loss', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(10, 110)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Plot saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare CITA_Instruct vs CITA_NoInstruct training progress',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two training runs
  python unit_test/compare_cita_training.py \\
      logs/CITA_Instruct_training_20251116_222112.log \\
      logs/CITA_NoInstruct_training_20251116_015222.log

  # Specify custom output path
  python unit_test/compare_cita_training.py \\
      logs/CITA_Instruct_training_20251116_222112.log \\
      logs/CITA_NoInstruct_training_20251116_015222.log \\
      --output logs_training/iter12/plots/custom_comparison.png
        """
    )

    parser.add_argument(
        'instruct_log',
        type=str,
        help='Path to CITA_Instruct training log'
    )

    parser.add_argument(
        'noinstruct_log',
        type=str,
        help='Path to CITA_NoInstruct training log'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='logs_training/iter12/plots/Instruct_vs_NoInstruct_progress.png',
        help='Output plot path (default: logs_training/iter12/plots/Instruct_vs_NoInstruct_progress.png)'
    )

    args = parser.parse_args()

    # Validate input files
    instruct_path = Path(args.instruct_log)
    noinstruct_path = Path(args.noinstruct_log)

    if not instruct_path.exists():
        print(f"❌ Error: CITA_Instruct log not found: {instruct_path}")
        sys.exit(1)

    if not noinstruct_path.exists():
        print(f"❌ Error: CITA_NoInstruct log not found: {noinstruct_path}")
        sys.exit(1)

    # Parse logs
    print("📊 Parsing training logs...")
    instruct_data = parse_training_log(instruct_path)
    noinstruct_data = parse_training_log(noinstruct_path)

    # Print summary
    print_summary(instruct_data, noinstruct_data)

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create plot
    print(f"\n📈 Generating comparison plot...")
    create_comparison_plot(instruct_data, noinstruct_data, output_path)

    print(f"\n✅ Comparison complete!")


if __name__ == '__main__':
    main()
