#!/usr/bin/env python3
"""
Simple Loss Curve Comparison - TensorBoard style
Extract and plot training loss curves from QLoRA and GRIT training
"""

import json
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def extract_qlora_losses():
    """Extract QLoRA training losses from trainer_state.json"""
    trainer_state_file = Path("outputs/m03a_qlora_training/checkpoints/checkpoint-250/trainer_state.json")

    if not trainer_state_file.exists():
        print("❌ QLoRA trainer_state.json not found")
        return [], []

    with open(trainer_state_file, 'r') as f:
        trainer_state = json.load(f)

    steps = []
    losses = []

    for entry in trainer_state.get('log_history', []):
        if 'loss' in entry and 'step' in entry:
            steps.append(entry['step'])
            losses.append(entry['loss'])

    print(f"✅ QLoRA: {len(steps)} training steps extracted")
    return steps, losses

def extract_grit_losses():
    """Extract GRIT training losses from log file"""
    log_file = Path("outputs/m03b_grit_training/grit_training.log")

    if not log_file.exists():
        print("❌ GRIT training log not found")
        return [], []

    steps = []
    losses = []

    with open(log_file, 'r') as f:
        lines = f.readlines()

    # Look for step-by-step loss values
    step_count = 0
    for line in lines:
        # Pattern: "Step X/Y: loss = Z.ZZZZ"
        step_match = re.search(r'Step (\d+)[/\d]*.*?loss[:\s=]+([0-9]+\.?[0-9]*)', line, re.IGNORECASE)
        if step_match:
            step_num = int(step_match.group(1))
            loss_val = float(step_match.group(2))
            steps.append(step_num)
            losses.append(loss_val)
            continue

        # Pattern: "loss: X.XXXX" or "Train Loss: X.XXXX"
        loss_match = re.search(r'(?:train\s+)?loss[:\s=]+([0-9]+\.?[0-9]*)', line, re.IGNORECASE)
        if loss_match:
            loss_val = float(loss_match.group(1))
            steps.append(step_count)
            losses.append(loss_val)
            step_count += 1

    print(f"✅ GRIT: {len(steps)} loss values extracted")
    return steps, losses

def create_loss_comparison_plot():
    """Create TensorBoard-style loss comparison plot"""

    # Extract data
    qlora_steps, qlora_losses = extract_qlora_losses()
    grit_steps, grit_losses = extract_grit_losses()

    if not qlora_steps and not grit_steps:
        print("❌ No training data found for either method")
        return

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Plot QLoRA
    if qlora_steps:
        plt.plot(qlora_steps, qlora_losses,
                label=f'QLoRA (final: {qlora_losses[-1]:.4f})',
                color='#1f77b4', linewidth=2, marker='o', markersize=4)
        print(f"📊 QLoRA: {len(qlora_steps)} steps, final loss: {qlora_losses[-1]:.4f}")

    # Plot GRIT
    if grit_steps:
        plt.plot(grit_steps, grit_losses,
                label=f'GRIT (final: {grit_losses[-1]:.4f})',
                color='#ff7f0e', linewidth=2, marker='s', markersize=4)
        print(f"📊 GRIT: {len(grit_steps)} steps, final loss: {grit_losses[-1]:.4f}")

    # Styling - TensorBoard look
    plt.title('Training Loss Comparison: QLoRA vs GRIT', fontsize=16, fontweight='bold')
    plt.xlabel('Training Steps', fontsize=14)
    plt.ylabel('Training Loss', fontsize=14)
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    plt.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)

    # Set background color similar to TensorBoard
    plt.gca().set_facecolor('#fafafa')
    plt.gcf().patch.set_facecolor('white')

    # Auto-adjust y limits for better visualization
    if qlora_losses or grit_losses:
        all_losses = qlora_losses + grit_losses
        y_min, y_max = min(all_losses), max(all_losses)
        y_range = y_max - y_min
        plt.ylim(y_min - y_range * 0.05, y_max + y_range * 0.05)

    plt.tight_layout()

    # Save the plot
    output_file = "outputs/m04_comparative_analysis/training_loss_comparison.png"
    Path("outputs/m04_comparative_analysis").mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"💾 Plot saved: {output_file}")

    # Show comparison summary
    print("\n" + "="*50)
    print("📈 TRAINING LOSS COMPARISON SUMMARY")
    print("="*50)

    if qlora_steps and grit_steps:
        print(f"QLoRA:  {qlora_losses[0]:.4f} → {qlora_losses[-1]:.4f} (Δ{qlora_losses[0]-qlora_losses[-1]:+.4f})")
        print(f"GRIT:   {grit_losses[0]:.4f} → {grit_losses[-1]:.4f} (Δ{grit_losses[0]-grit_losses[-1]:+.4f})")

        if qlora_losses[-1] < grit_losses[-1]:
            print(f"🏆 QLoRA achieved better final loss by {grit_losses[-1]-qlora_losses[-1]:.4f}")
        else:
            print(f"🏆 GRIT achieved better final loss by {qlora_losses[-1]-grit_losses[-1]:.4f}")

    plt.show()

if __name__ == "__main__":
    print("🔬 Simple Loss Curve Comparison")
    print("="*40)
    create_loss_comparison_plot()