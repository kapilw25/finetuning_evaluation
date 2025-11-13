"""
Check if Optuna learned from Trial 0 explosion
"""
import sqlite3
import pandas as pd

db_path = "outputs/optuna_cita.db"
conn = sqlite3.connect(db_path)

# Get trial states
print("=" * 80)
print("TRIAL STATES")
print("=" * 80)
trials_df = pd.read_sql_query("""
    SELECT trial_id, state, datetime_start, datetime_complete
    FROM trials
    ORDER BY trial_id
""", conn)
print(trials_df.to_string(index=False))

# Get hyperparameters for each trial
print("\n" + "=" * 80)
print("HYPERPARAMETERS BY TRIAL")
print("=" * 80)

params_df = pd.read_sql_query("""
    SELECT tp.trial_id, tp.param_name, tp.param_value
    FROM trial_params tp
    ORDER BY tp.trial_id, tp.param_name
""", conn)

# Pivot to wide format
params_wide = params_df.pivot(index='trial_id', columns='param_name', values='param_value')
print(params_wide.to_string())

# Check if consecutive trials have same hyperparameters
print("\n" + "=" * 80)
print("COMPARISON: Consecutive Trials")
print("=" * 80)

trial_ids = sorted(params_wide.index.tolist())
print(f"\nAvailable trials: {trial_ids}")

if len(trial_ids) >= 2:
    for i in range(len(trial_ids) - 1):
        trial_a_id = trial_ids[i]
        trial_b_id = trial_ids[i + 1]

        trial_a = params_wide.loc[trial_a_id]
        trial_b = params_wide.loc[trial_b_id]

        print(f"\n--- Trial {trial_a_id} vs Trial {trial_b_id} ---")

        if trial_a.equals(trial_b):
            print(f"❌ IDENTICAL! Optuna is RE-TRYING same hyperparameters")
            print(f"   This happens because:")
            print(f"   1. TPESampler's n_startup_trials=10 (first 10 are RANDOM)")
            print(f"   2. Explosion (TrialPruned) does NOT provide objective values")
            print(f"   3. TPE cannot 'learn' without objective values")
        else:
            print(f"✅ DIFFERENT! Optuna sampled new hyperparameters")

            for param in trial_a.index:
                val_a = trial_a[param]
                val_b = trial_b[param]
                if val_a != val_b:
                    diff_pct = abs(val_b - val_a) / val_a * 100
                    print(f"   {param}: {val_a:.6f} → {val_b:.6f} ({diff_pct:+.1f}%)")

# Check trial values (objectives)
print("\n" + "=" * 80)
print("TRIAL OBJECTIVES (margin, accuracy, -chosen)")
print("=" * 80)

values_df = pd.read_sql_query("""
    SELECT trial_id, value_id, value
    FROM trial_values
    ORDER BY trial_id, value_id
""", conn)

if len(values_df) > 0:
    values_wide = values_df.pivot(index='trial_id', columns='value_id', values='value')
    values_wide.columns = ['margin', 'accuracy', 'neg_chosen']
    print(values_wide.to_string())
else:
    print("No completed trials with valid objectives")

conn.close()
