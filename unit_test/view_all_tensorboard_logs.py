#!/usr/bin/env python3
"""
View all TensorBoard logs from multiple training runs
"""

from tensorboard.backend.event_processing import event_accumulator
import os
import glob

tensorboard_base = "/home/ubuntu/DiskUsEast1/finetuning_evaluation/tensorboard_logs"

# Find all run directories
run_dirs = sorted(glob.glob(os.path.join(tensorboard_base, "*")))

print("="*100)
print("ALL TENSORBOARD LOGS SUMMARY")
print("="*100)

for run_dir in run_dirs:
    run_name = os.path.basename(run_dir)

    print(f"\n{'='*100}")
    print(f"Run: {run_name}")
    print(f"{'='*100}")

    try:
        ea = event_accumulator.EventAccumulator(run_dir)
        ea.Reload()

        # Get all scalar tags
        tags = ea.Tags()['scalars']

        if not tags:
            print("  ⚠️  No scalar data found")
            continue

        # Categorize tags
        loss_tags = [t for t in tags if 'loss' in t.lower()]
        cita_tags = [t for t in tags if 'cita' in t.lower()]
        reward_tags = [t for t in tags if 'reward' in t.lower()]
        other_tags = [t for t in tags if t not in loss_tags and t not in cita_tags and t not in reward_tags]

        print(f"\n📊 Total metrics: {len(tags)}")

        # Show loss metrics
        if loss_tags:
            print(f"\n🔴 Loss Metrics ({len(loss_tags)}):")
            for tag in sorted(loss_tags):
                values = ea.Scalars(tag)
                if values:
                    first_val = values[0].value
                    last_val = values[-1].value
                    print(f"  {tag:45s} | Steps: {len(values):4d} | First: {first_val:10.6f} | Last: {last_val:10.6f}")

        # Show CITA-specific metrics
        if cita_tags:
            print(f"\n🟢 CITA Metrics ({len(cita_tags)}):")
            for tag in sorted(cita_tags):
                values = ea.Scalars(tag)
                if values:
                    first_val = values[0].value
                    last_val = values[-1].value
                    print(f"  {tag:45s} | Steps: {len(values):4d} | First: {first_val:10.6f} | Last: {last_val:10.6f}")

        # Show reward metrics
        if reward_tags:
            print(f"\n🟡 Reward Metrics ({len(reward_tags)}):")
            for tag in sorted(reward_tags):
                values = ea.Scalars(tag)
                if values:
                    first_val = values[0].value
                    last_val = values[-1].value
                    print(f"  {tag:45s} | Steps: {len(values):4d} | First: {first_val:10.6f} | Last: {last_val:10.6f}")

        # Show other metrics
        if other_tags:
            print(f"\n⚪ Other Metrics ({len(other_tags)}):")
            for tag in sorted(other_tags):
                values = ea.Scalars(tag)
                if values:
                    first_val = values[0].value
                    last_val = values[-1].value
                    print(f"  {tag:45s} | Steps: {len(values):4d} | First: {first_val:10.6f} | Last: {last_val:10.6f}")

        # Training progress summary
        if 'train/loss' in tags:
            train_loss = ea.Scalars('train/loss')
            print(f"\n📈 Training Progress:")
            print(f"  Total steps: {len(train_loss)}")
            print(f"  Initial loss: {train_loss[0].value:.6f}")
            print(f"  Final loss: {train_loss[-1].value:.6f}")
            print(f"  Loss change: {train_loss[-1].value - train_loss[0].value:+.6f}")

    except Exception as e:
        print(f"  ❌ Error reading logs: {e}")

print("\n" + "="*100)
print("✅ Summary Complete")
print("="*100)
