#!/usr/bin/env python3
"""
Extract and compare Optuna trial hyperparameters, eval metrics, and training metrics
Usage:
1. Multiple Input Modes:
# Default: All logs in logs/ directory
python3 unit_test/optuna_trial_comparison.py

# Single file
python3 unit_test/optuna_trial_comparison.py logs/CITA_Adaptive_training_20251115_042538.log

# Multiple files
python3 unit_test/optuna_trial_comparison.py file1.log file2.log file3.log
"""
import re
import sys
from pathlib import Path
from glob import glob

def extract_trial_data(log_file):
    """Extract all trial data from Optuna log file"""

    with open(log_file, 'r') as f:
        log_content = f.read()

    # Find all trial boundaries
    trial_starts = [(m.start(), int(m.group(1))) for m in re.finditer(r'🔬 Trial (\d+): Testing hyperparameters', log_content)]

    trials = {}
    log_filename = Path(log_file).name

    for idx, (start_pos, trial_num) in enumerate(trial_starts):
        # Determine end position (start of next trial or end of file)
        end_pos = trial_starts[idx + 1][0] if idx + 1 < len(trial_starts) else len(log_content)
        trial_section = log_content[start_pos:end_pos]

        trials[trial_num] = {
            'hyperparameters': extract_hyperparameters(trial_section),
            'eval_metrics': extract_eval_metrics(trial_section),
            'training_summary': extract_training_summary(trial_section),
            'status': extract_status(trial_section),
            'log_file': log_filename  # Track which log file this trial came from
        }

    return trials

def extract_hyperparameters(section):
    """Extract hyperparameters from trial section"""
    hp = {}

    patterns = {
        'lambda_kl': r'lambda_kl:\s+([\d.e-]+)',
        'learning_rate': r'learning_rate:\s+([\d.e-]+)',
        'beta': r'beta:\s+([\d.]+)',
        'weight_decay': r'weight_decay:\s+([\d.]+)',
        'warmup_ratio': r'warmup_ratio:\s+([\d.]+)'
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, section)
        if match:
            hp[key] = float(match.group(1))

    return hp

def extract_eval_metrics(section):
    """Extract all eval checkpoint metrics"""
    eval_pattern = r"'eval_loss':\s+([\d.]+).*?'eval_rewards/accuracies':\s+([\d.]+).*?'eval_rewards/margins':\s+([\d.]+).*?'epoch':\s+([\d.]+)"

    evals = []
    for match in re.finditer(eval_pattern, section):
        evals.append({
            'eval_loss': float(match.group(1)),
            'eval_accuracy': float(match.group(2)),
            'eval_margin': float(match.group(3)),
            'epoch': float(match.group(4))
        })

    return evals

def extract_training_summary(section):
    """Extract key training metrics summary"""
    summary = {}

    # Find first and last training summaries
    summaries = list(re.finditer(r'📊 TRAINING SUMMARY - Step (\d+).*?(?=📊 TRAINING SUMMARY|$)', section, re.DOTALL))

    if summaries:
        first_summary = summaries[0].group(0)
        last_summary = summaries[-1].group(0)

        # Extract from first summary
        step_match = re.search(r'Step (\d+)', first_summary)
        loss_match = re.search(r'Total loss:.*?\(avg:\s+([\d.-]+)\)', first_summary)
        kl_match = re.search(r'L_KL:.*?\(avg:\s+(-?[\d.]+)\)', first_summary)
        neg_match = re.search(r'Negative samples:.*?\((\d+)%\)', first_summary)
        acc_match = re.search(r'Average:\s+(\d+)%', first_summary)

        summary['first_step'] = int(step_match.group(1)) if step_match else None
        summary['first_loss'] = float(loss_match.group(1)) if loss_match else None
        summary['first_kl'] = float(kl_match.group(1)) if kl_match else None
        summary['first_neg_margins'] = int(neg_match.group(1)) if neg_match else None
        summary['first_accuracy'] = int(acc_match.group(1)) if acc_match else None

        # Extract from last summary
        step_match = re.search(r'Step (\d+)', last_summary)
        loss_match = re.search(r'Total loss:.*?\(avg:\s+([\d.-]+)\)', last_summary)
        kl_match = re.search(r'L_KL:.*?\(avg:\s+(-?[\d.]+)\)', last_summary)
        neg_match = re.search(r'Negative samples:.*?\((\d+)%\)', last_summary)
        acc_match = re.search(r'Average:\s+(\d+)%', last_summary)

        summary['last_step'] = int(step_match.group(1)) if step_match else None
        summary['last_loss'] = float(loss_match.group(1)) if loss_match else None
        summary['last_kl'] = float(kl_match.group(1)) if kl_match else None
        summary['last_neg_margins'] = int(neg_match.group(1)) if neg_match else None
        summary['last_accuracy'] = int(acc_match.group(1)) if acc_match else None

    return summary

def extract_status(section):
    """Determine trial status"""
    if 'EXPLOSION DETECTED' in section:
        match = re.search(r'grad_norm=([\d.]+)', section)
        grad_norm = match.group(1) if match else 'unknown'
        return f'EXPLODED (grad_norm={grad_norm})'
    elif '1350/1354' in section or '1354/1354' in section:
        return 'COMPLETED'
    else:
        # Find current progress (last occurrence)
        progress_matches = list(re.finditer(r'(\d+)/1354', section))
        if progress_matches:
            step = int(progress_matches[-1].group(1))
            pct = int(step / 1354 * 100)
            return f'IN_PROGRESS (step {step}, {pct}%)'
        return 'UNKNOWN'

def print_hyperparameters_table(trials):
    """Print hyperparameters comparison table"""
    print("\n" + "="*160)
    print("HYPERPARAMETERS COMPARISON")
    print("="*160)

    headers = ['Trial', 'lambda_kl', 'learning_rate', 'beta', 'weight_decay', 'warmup_ratio', 'Status', 'Log File']

    # Print header
    print(f"{headers[0]:<8} {headers[1]:<12} {headers[2]:<15} {headers[3]:<8} {headers[4]:<14} {headers[5]:<14} {headers[6]:<30} {headers[7]}")
    print("-"*160)

    # Print each trial
    for trial_num in sorted(trials.keys()):
        hp = trials[trial_num]['hyperparameters']
        status = trials[trial_num]['status']
        log_file = trials[trial_num].get('log_file', 'N/A')
        # Shorten log filename for display
        log_display = log_file.replace('CITA_Adaptive_training_', '').replace('.log', '') if log_file != 'N/A' else 'N/A'

        print(f"{trial_num:<8} "
              f"{hp.get('lambda_kl', 0):<12.6f} "
              f"{hp.get('learning_rate', 0):<15.6e} "
              f"{hp.get('beta', 0):<8.4f} "
              f"{hp.get('weight_decay', 0):<14.4f} "
              f"{hp.get('warmup_ratio', 0):<14.4f} "
              f"{status:<30} "
              f"{log_display}")

    print()

def print_eval_metrics_table(trials):
    """Print eval metrics comparison at each 20% checkpoint"""
    print("\n" + "="*120)
    print("EVAL METRICS @ EACH 20% CHECKPOINT")
    print("="*120)

    for trial_num in sorted(trials.keys()):
        evals = trials[trial_num]['eval_metrics']
        status = trials[trial_num]['status']

        print(f"\nTrial {trial_num} - {status}")
        print("-"*80)

        if evals:
            print(f"{'Epoch':<8} {'%':<6} {'eval_loss':<12} {'eval_accuracy':<15} {'eval_margin':<12}")
            print("-"*80)

            for eval_data in evals:
                epoch = eval_data['epoch']
                pct = int(epoch * 100)
                print(f"{epoch:<8.1f} {pct:<6}% "
                      f"{eval_data['eval_loss']:<12.4f} "
                      f"{eval_data['eval_accuracy']*100:<14.1f}% "
                      f"{eval_data['eval_margin']:<12.2f}")
        else:
            print("  No eval checkpoints yet")

    print()

def print_training_summary_table(trials):
    """Print training metrics summary (first vs last)"""
    print("\n" + "="*120)
    print("TRAINING METRICS SUMMARY (First Step vs Last Step)")
    print("="*120)

    for trial_num in sorted(trials.keys()):
        summary = trials[trial_num]['training_summary']
        status = trials[trial_num]['status']

        if not summary:
            continue

        print(f"\nTrial {trial_num} - {status}")
        print("-"*80)

        first_step = summary.get('first_step', 'N/A')
        last_step = summary.get('last_step', 'N/A')
        print(f"{'Metric':<20} {'First (Step ' + str(first_step) + ')':<25} {'Last (Step ' + str(last_step) + ')':<25} {'Change'}")
        print("-"*80)

        # Loss
        first_loss = summary.get('first_loss')
        last_loss = summary.get('last_loss')
        if first_loss and last_loss:
            change = ((last_loss - first_loss) / abs(first_loss)) * 100 if first_loss != 0 else 0
            print(f"{'Avg Loss':<20} {first_loss:<25.4f} {last_loss:<25.4f} {change:+.1f}%")

        # L_KL
        first_kl = summary.get('first_kl')
        last_kl = summary.get('last_kl')
        if first_kl is not None and last_kl is not None:
            change = last_kl - first_kl
            print(f"{'Avg L_KL':<20} {first_kl:<25.2f} {last_kl:<25.2f} {change:+.2f}")

        # Negative margins
        first_neg = summary.get('first_neg_margins')
        last_neg = summary.get('last_neg_margins')
        if first_neg is not None and last_neg is not None:
            change = last_neg - first_neg
            print(f"{'Negative margins %':<20} {first_neg:<25}% {last_neg:<25}% {change:+d}%")

        # Accuracy
        first_acc = summary.get('first_accuracy')
        last_acc = summary.get('last_accuracy')
        if first_acc is not None and last_acc is not None:
            change = last_acc - first_acc
            print(f"{'Accuracy %':<20} {first_acc:<25}% {last_acc:<25}% {change:+d}%")

    print()

def print_eval_comparison_at_20pct(trials):
    """Print side-by-side comparison of all trials at 20% checkpoint"""
    print("\n" + "="*160)
    print("COMPARISON @ 20% CHECKPOINT")
    print("="*160)

    print(f"{'Trial':<8} {'eval_loss':<12} {'eval_accuracy':<15} {'eval_margin':<12} {'Status':<30} {'Log File'}")
    print("-"*160)

    for trial_num in sorted(trials.keys()):
        evals = trials[trial_num]['eval_metrics']
        status = trials[trial_num]['status']
        log_file = trials[trial_num].get('log_file', 'N/A')
        log_display = log_file.replace('CITA_Adaptive_training_', '').replace('.log', '') if log_file != 'N/A' else 'N/A'

        # Find 20% checkpoint (epoch 0.2)
        eval_20 = next((e for e in evals if e['epoch'] == 0.2), None)

        if eval_20:
            print(f"{trial_num:<8} "
                  f"{eval_20['eval_loss']:<12.4f} "
                  f"{eval_20['eval_accuracy']*100:<14.1f}% "
                  f"{eval_20['eval_margin']:<12.2f} "
                  f"{status:<30} "
                  f"{log_display}")
        else:
            print(f"{trial_num:<8} {'N/A':<12} {'N/A':<15} {'N/A':<12} {status:<30} {log_display}")

    print()

def get_log_files():
    """Get log files from command line arguments"""
    if len(sys.argv) == 1:
        # No arguments - use default directory
        log_dir = "/lambda/nfs/DiskUsEast1/finetuning_evaluation/logs"
        pattern = f"{log_dir}/CITA_Adaptive_training_*.log"
        files = sorted(glob(pattern))
        if not files:
            print(f"Error: No log files found matching {pattern}")
            sys.exit(1)
        return files

    elif sys.argv[1] == '--dir':
        # Directory specified
        if len(sys.argv) < 3:
            print("Error: --dir requires a directory path")
            sys.exit(1)
        log_dir = sys.argv[2]
        pattern = f"{log_dir}/CITA_Adaptive_training_*.log"
        files = sorted(glob(pattern))
        if not files:
            print(f"Error: No log files found matching {pattern}")
            sys.exit(1)
        return files

    else:
        # File(s) specified
        files = sys.argv[1:]
        for f in files:
            if not Path(f).exists():
                print(f"Error: Log file not found: {f}")
                sys.exit(1)
        return files

def main():
    log_files = get_log_files()

    print(f"Parsing {len(log_files)} log file(s):")
    for f in log_files:
        print(f"  - {Path(f).name}")

    # Extract trials from all log files
    all_trials = {}
    for log_file in log_files:
        trials = extract_trial_data(log_file)
        # Merge trials (trial IDs should be unique across all logs)
        for trial_num, trial_data in trials.items():
            if trial_num in all_trials:
                print(f"\n⚠️  Warning: Trial {trial_num} found in multiple logs!")
                print(f"    First:  {all_trials[trial_num]['log_file']}")
                print(f"    Second: {trial_data['log_file']}")
                print(f"    Using data from: {all_trials[trial_num]['log_file']}")
            else:
                all_trials[trial_num] = trial_data

    print(f"\nFound {len(all_trials)} unique trials across all logs")

    # Print all tables (using consolidated trials)
    print_hyperparameters_table(all_trials)
    print_eval_comparison_at_20pct(all_trials)
    print_eval_metrics_table(all_trials)
    print_training_summary_table(all_trials)

    print("\n" + "="*120)
    print("SUMMARY")
    print("="*120)

    completed = sum(1 for t in all_trials.values() if 'COMPLETED' in t['status'])
    exploded = sum(1 for t in all_trials.values() if 'EXPLODED' in t['status'])
    in_progress = sum(1 for t in all_trials.values() if 'IN_PROGRESS' in t['status'])

    print(f"Total trials: {len(all_trials)}")
    print(f"  Completed: {completed}")
    print(f"  Exploded: {exploded}")
    print(f"  In progress: {in_progress}")

    # Show log file breakdown
    log_counts = {}
    for trial_data in all_trials.values():
        log_file = trial_data['log_file']
        log_counts[log_file] = log_counts.get(log_file, 0) + 1

    print(f"\nTrials per log file:")
    for log_file in sorted(log_counts.keys()):
        print(f"  {log_file}: {log_counts[log_file]} trials")

    print("="*120)

if __name__ == "__main__":
    main()
