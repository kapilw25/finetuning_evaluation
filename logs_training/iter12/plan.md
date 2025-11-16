Implementation Plan: Full Ablation (6 Models) for Tier 1 Paper
>> every name should have suffix either "*_Instruct" or "*_NoInstruct"

   ===================================================================
   
   Phase 1: Code Modifications (3-4 hours)
   
   ===================================================================

   TODO Progress:
   - ✅ formatters.py: Add format_sft_Instruct + format_dpo_Instruct + aliases
   - ✅ model_utils.py: Rename 3 keys (SFT_Baseline→SFT_NoInstruct, DPO_Baseline→DPO_NoInstruct, CITA_Baseline→CITA_Instruct)
   - ✅ model_utils.py: Add 3 new MODEL_NAME_MAP entries (SFT_Instruct, DPO_Instruct, CITA_NoInstruct)
   - ✅ SFT script: Add USE_INSTRUCTION=False toggle + RUN_NAME conditional + formatter conditional + output_dir
   - ✅ DPO script: Add USE_INSTRUCTION=False toggle + RUN_NAME conditional + formatter conditional + output_dir
   - ✅ CITA script: Fix USE_INSTRUCTIONS→USE_INSTRUCTION + RUN_NAME conditional
   - ✅ CITA script: Replace Trial 2 with Trial 5 hyperparameters
   - ✅ toxicity.py: Expand MODELS to 7 variants + rename 3 keys + add use_instruction logic

   ---
   Step 1.1: Create new formatters in comparative_study/0c_utils/data_prep/formatters.py
   - Add format_sft_instruct() - SFT with system instruction
   - Add format_dpo_instruct() - DPO with system instruction
   - Add aliases: format_pku_for_sft_instruct, format_pku_for_dpo_instruct
   - Note: format_pku_for_cita_no_instruct already exists as alias to format_dpo (line 116)

   Step 1.2: Modify SFT script comparative_study/01a_SFT_Baseline/Llama3_BF16.py
   - Add `USE_INSTRUCTION = False` toggle (line ~80)
   - Update RUN_NAME to use conditional naming: "SFT_Instruct" if USE_INSTRUCTION else "SFT_NoInstruct"
   - Replace hardcoded formatter with conditional: format_pku_for_sft_instruct if USE_INSTRUCTION else format_pku_for_sft
   (line ~197)
   - Update formatting description to show instruction status

   Step 1.3: Modify DPO script comparative_study/02a_DPO_Baseline/Llama3_BF16.py
   - Add `USE_INSTRUCTION = False` toggle (line ~101)
   - Update RUN_NAME to use conditional naming: "DPO_Instruct" if USE_INSTRUCTION else "DPO_NoInstruct"
   - Replace hardcoded formatter with conditional: format_pku_for_dpo_instruct if USE_INSTRUCTION else format_pku_for_dpo
   (line ~235)
   - Update formatting description to show instruction status

   Step 1.4: Modify CITA script comparative_study/03a_CITA_Baseline/Llama3_BF16.py
   - Update Trial 5 hyperparameters (replace Trial 2 values at lines 130-136):
     - LAMBDA_KL = 0.000520
     - LEARNING_RATE = 6.827978e-06
     - BETA = 0.1191
     - WEIGHT_DECAY = 0.0091
     - WARMUP_RATIO = 0.0749
   - Fix variable name: USE_INSTRUCTIONS → USE_INSTRUCTION (line 109)
   - Update RUN_NAME to use conditional naming: "CITA_Instruct" if USE_INSTRUCTION else "CITA_NoInstruct"
   - Formatter already uses conditional logic (line 231)

   Step 1.5: Update model registry comparative_study/0c_utils/model_utils.py
   - Rename 3 existing keys: SFT_Baseline→SFT_NoInstruct, DPO_Baseline→DPO_NoInstruct, CITA_Baseline→CITA_Instruct
   - Add 3 new entries to MODEL_NAME_MAP (around line 548):
     - "SFT_Instruct": "kapilw25/llama3-8b-pku-SFT-Instruct-Baseline-NoInstruct"
     - "DPO_Instruct": "kapilw25/llama3-8b-pku-DPO-Instruct-SFT-Instruct"
     - "CITA_NoInstruct": "kapilw25/llama3-8b-pku-CITA-NoInstruct-DPO-NoInstruct"

   Step 1.6: Update toxicity eval comparative_study/05_evaluation/llm_as_judge/toxicity.py
   - Expand MODELS dict to include all 6 variants (lines 52-69)
   - Add "use_instruction": bool field to each model config
   - Update inference logic to use model_config["use_instruction"] instead of hardcoded model_key check (lines 305-314)

   ===================================================================
   Phase 2: Training All 6 Variants
   ===================================================================

   Training Progress (as of 2025-11-16):

   ✅ Completed (3/6 models):
   1. SFT_NoInstruct - outputs/SFT_NoInstruct ✓
   2. SFT_Instruct - outputs/SFT_Instruct ✓
   3. DPO_NoInstruct - outputs/DPO_NoInstruct ✓

   🔄 In Progress (1/6 models):
   4. CITA_NoInstruct - outputs/CITA_NoInstruct (15% complete, ~76 min remaining)

   ⏳ Remaining (2/6 models):
   5. DPO_Instruct - Need to train (~103 minutes)
   6. CITA_Instruct - Need to train (~120 minutes)

   Total Time Remaining: ~3.7 hours (with parallel execution on 2 instances)

   ---
   COMMAND-LINE FLAG: --use-instruction {true|false} (REQUIRED)
   - All training scripts now require --use-instruction flag
   - No manual file editing needed
   - Enables parallel training on separate instances

   ---
   INSTANCE 1: NoInstruct Chain (run sequentially: SFT → DPO → CITA)

   1. SFT_NoInstruct
   ```
   cd /lambda/nfs/DiskUsEast1/finetuning_evaluation
   python comparative_study/01a_SFT_Baseline/Llama3_BF16.py --mode full --use-instruction false
   ```

   Expected HF repo: kapilw25/llama3-8b-pku-SFT-NoInstruct-Baseline-NoInstruct

   2. DPO_NoInstruct (wait for SFT_NoInstruct to complete)
   ```
   cd /lambda/nfs/DiskUsEast1/finetuning_evaluation
   python comparative_study/02a_DPO_Baseline/Llama3_BF16.py --mode full --use-instruction false --base_model kapilw25/llama3-8b-pku-SFT-NoInstruct-Baseline-NoInstruct
   ```

   Expected HF repo: kapilw25/llama3-8b-pku-DPO-NoInstruct-SFT-NoInstruct

   3. CITA_NoInstruct (wait for DPO_NoInstruct to complete)
   ```
   cd /lambda/nfs/DiskUsEast1/finetuning_evaluation
   python comparative_study/03a_CITA_Baseline/Llama3_BF16.py --mode full --use-instruction false --base_model kapilw25/llama3-8b-pku-DPO-NoInstruct-SFT-NoInstruct
   ```

   Expected HF repo: kapilw25/llama3-8b-pku-CITA-NoInstruct-DPO-NoInstruct

   ---
   INSTANCE 2: Instruct Chain (run sequentially: SFT → DPO → CITA)

   1. SFT_Instruct
   ```
   cd /lambda/nfs/DiskUsEast1/finetuning_evaluation
   python comparative_study/01a_SFT_Baseline/Llama3_BF16.py --mode full --use-instruction true
   ```

   Expected HF repo: kapilw25/llama3-8b-pku-SFT-Instruct-Baseline-NoInstruct

   2. DPO_Instruct (wait for SFT_Instruct to complete)
   ```
   cd /lambda/nfs/DiskUsEast1/finetuning_evaluation
   python comparative_study/02a_DPO_Baseline/Llama3_BF16.py --mode full --use-instruction true --base_model kapilw25/llama3-8b-pku-SFT-Instruct-Baseline-NoInstruct
   ```

   Expected HF repo: kapilw25/llama3-8b-pku-DPO-Instruct-SFT-Instruct

   3. CITA_Instruct (wait for DPO_Instruct to complete, with Trial 5 HPs)
   ```
   cd /lambda/nfs/DiskUsEast1/finetuning_evaluation
   python comparative_study/03a_CITA_Baseline/Llama3_BF16.py --mode full --use-instruction true --base_model kapilw25/llama3-8b-pku-DPO-Instruct-SFT-Instruct
   ```

   Expected HF repo: kapilw25/llama3-8b-pku-CITA-Instruct-DPO-Instruct

   ===================================================================
   Phase 3: Evaluation (~2 hours)
   ===================================================================

   TODO Progress:
   ⏳ Run toxicity evaluation on all 7 models (Baseline + 6 variants)
   ⏳ Create statistical analysis script (ANOVA, t-tests, effect sizes)
   ⏳ Generate comparison plots and tables

   ---
   Step 3.1: Run toxicity evaluation

   Available modes (run sequentially):

   3.1a. Sanity check (150 samples, ~15 min)
   ```
   python comparative_study/05_evaluation/llm_as_judge/toxicity.py --mode sanity
   ```

   3.1b. Full evaluation (3,684 samples, ~60 min)
   ```
   python comparative_study/05_evaluation/llm_as_judge/toxicity.py --mode full
   ```

   3.1c. Custom sample count (optional)
   ```
   python comparative_study/05_evaluation/llm_as_judge/toxicity.py --toxicity-samples 100
   ```

   Evaluates all 7 models: Baseline + SFT/DPO/CITA x NoInstruct/Instruct

   Step 3.2: Statistical analysis
   - Create analysis script for:
     - 2-way ANOVA: Method (SFT/DPO/CITA) x Instruction (Yes/No)
     - Pairwise t-tests with Bonferroni correction
     - Effect sizes (Cohen's d)
     - Confidence intervals

   ===================================================================
   Phase 4: Tier 1 Enhancements (4-6 hours) - FUTURE WORK
   ===================================================================

   TODO Progress:
   ⏳ Implement helpfulness evaluation
   ⏳ Add instruction-following accuracy metric
   ⏳ Create out-of-distribution test set
   ⏳ Compare against GPT-4, Claude-3 baselines
   ⏳ Run benchmarks (ToxiGen, RealToxicityPrompts)

   ---
   Step 4.1: Additional evaluation metrics (FUTURE WORK - document as limitation if time-constrained)
   - Helpfulness evaluation
   - Instruction-following accuracy
   - Out-of-distribution test set

   Step 4.2: Baseline comparisons (FUTURE WORK)
   - Compare against GPT-4, Claude-3 on same test set
   - Run on ToxiGen, RealToxicityPrompts benchmarks

   ---
   Expected Results Table

   | Model           | Train Instruct | Test Instruct | Toxicity (lower) | Refusal Rate (higher) |
   |-----------------|----------------|---------------|------------------|-----------------------|
   | SFT_NoInstruct  | No             | No            | ?                | ?                     |
   | SFT_Instruct    | Yes            | Yes           | ?                | ?                     |
   | DPO_NoInstruct  | No             | No            | baseline         | baseline              |
   | DPO_Instruct    | Yes            | Yes           | ?                | ?                     |
   | CITA_NoInstruct | No             | No            | ?                | ?                     |
   | CITA_Instruct   | Yes            | Yes           | best?            | best?                 |

   Research Questions:
   1. Instruction Effect: Does instruction help? (Compare NoInstruct vs Instruct within each method)
   2. Method Effect: Which method is best? (Compare SFT vs DPO vs CITA within each instruction setting)
   3. Interaction: Does instruction help more for CITA than SFT/DPO?

   ---
   Total Effort Estimate

   - Code modifications: 3-4 hours
   - Training: 10-12 GPU hours
   - Evaluation: 2 hours
   - Analysis: 4 hours
   - Total: 1-2 days work + 12 GPU hours

   ---
   Files to Modify

   1. comparative_study/0c_utils/data_prep/formatters.py (add 2 formatters + 2 aliases)
   2. comparative_study/01a_SFT_Baseline/Llama3_BF16.py (add toggle + rename RUN_NAME)
   3. comparative_study/02a_DPO_Baseline/Llama3_BF16.py (add toggle + rename RUN_NAME)
   4. comparative_study/03a_CITA_Baseline/Llama3_BF16.py (update Trial 5 HPs + rename RUN_NAME + fix variable)
   5. comparative_study/0c_utils/model_utils.py (rename 3 keys + add 3 new entries)
   6. comparative_study/05_evaluation/llm_as_judge/toxicity.py (expand to 7 models + rename 3 keys)

   ===================================================================
   DETAILED IMPLEMENTATION GUIDE (Code-Level Changes)
   ===================================================================

   Trial 5 Hyperparameters (from Optuna - Selected for STABILITY):

   IMPORTANT: Trial 5 was NOT the best performer, but was selected for:
   - Most STABLE training (smooth loss/accuracy/margin curves)
   - Good balance across all metrics (loss, accuracy, margin)
   - Trial 0 had better raw metrics but UNSTABLE training (see tensorboard_2.png)

   Hyperparameters:
   lambda_kl:       0.000520
   learning_rate:   6.827978e-06
   beta:            0.1191
   weight_decay:    0.0091
   warmup_ratio:    0.0749

   Final metrics (epoch 1.0):
   eval_loss:       0.2791
   eval_accuracy:   89.5%
   eval_margin:     6.95

   Transferability to CITA_Instruct:
   ✅ PROS (why Trial 5 HPs should work):
   - Same base architecture (Llama-3.1-8B)
   - Same dataset (PKU-SafeRLHF, only formatting differs)
   - Showed excellent stability on NoInstruct variant
   - No time for separate Optuna search (~59 hours)

   ⚠️  RISKS (why they might NOT work):
   - System instructions change loss landscape
   - Instruct formatting → longer responses → different gradient magnitudes
   - No empirical validation of HP transfer across instruction modes

   📋 MONITORING PLAN:
   - Watch first 100 steps for NaN/Inf errors
   - Check margin stability (should stay positive and increasing)
   - If training fails: execute Optuna search for CITA_Instruct (typically finds stable HPs within 5 trials)

   ---
   📁 FORMATTERS (comparative_study/0c_utils/data_prep/formatters.py)

   Add 2 new formatters (after line 108):

   ```python
   def format_sft_instruct(example: Dict) -> Dict:
       """SFT with system instruction (CITA-style)"""
       safe_response, _, harmful_categories = get_safe_unsafe_responses(example)
       instruction = synthesize_system_instruction(harmful_categories)

       messages = [
           {"role": "system", "content": instruction},
           {"role": "user", "content": example['prompt']},
           {"role": "assistant", "content": safe_response}
       ]
       return {"messages": messages}


   def format_dpo_instruct(example: Dict) -> Dict:
       """DPO with system instruction (CITA-style)"""
       safe_response, unsafe_response, harmful_categories = get_safe_unsafe_responses(example)
       instruction = synthesize_system_instruction(harmful_categories)

       prompt_messages = [
           {"role": "system", "content": instruction},
           {"role": "user", "content": example['prompt']}
       ]

       chosen_messages = [{"role": "assistant", "content": safe_response}]
       rejected_messages = [{"role": "assistant", "content": unsafe_response}]

       return {
           "prompt": prompt_messages,
           "chosen": chosen_messages,
           "rejected": rejected_messages,
       }
   ```

   Add aliases (after line 116):

   ```python
   format_pku_for_sft_instruct = format_sft_instruct
   format_pku_for_dpo_instruct = format_dpo_instruct
   ```

   ---
   📁 SFT SCRIPT (comparative_study/01a_SFT_Baseline/Llama3_BF16.py)

   Line 80: Add instruction toggle

   ```python
   # ===================================================================
   # INSTRUCTION MODE TOGGLE
   # ===================================================================
   `USE_INSTRUCTION = False`  # False: SFT_NoInstruct, True: SFT_Instruct
   ```

   Line 80-82: Update RUN_NAME (REVISED NAMING)

   ```python
   RUN_NAME = "SFT_Instruct" if USE_INSTRUCTION else "SFT_NoInstruct"
   HF_REPO = get_model_repo_name(RUN_NAME, precision="bf16")
   ```

   Line 197-201: Replace formatter import and usage

   ```python
   from data_prep.formatters import format_pku_for_sft, format_pku_for_sft_instruct

   # Format for SFT (conditional: WITH or WITHOUT instruction)
   formatter = format_pku_for_sft_instruct if USE_INSTRUCTION else format_pku_for_sft
   train_dataset = dataset_split['train'].map(
       formatter,
       remove_columns=dataset_split['train'].column_names,
       desc=f"Formatting PKU for SFT ({'WITH' if USE_INSTRUCTION else 'NO'} instruction)"
   )

   val_dataset = dataset_split['test'].map(
       formatter,
       remove_columns=dataset_split['test'].column_names,
       desc=f"Formatting PKU validation for SFT ({'WITH' if USE_INSTRUCTION else 'NO'} instruction)"
   )
   ```

   ---
   📁 DPO SCRIPT (comparative_study/02a_DPO_Baseline/Llama3_BF16.py)

   Line 101: Add instruction toggle

   ```python
   # ===================================================================
   # INSTRUCTION MODE TOGGLE
   # ===================================================================
   `USE_INSTRUCTION = False`  # False: DPO_NoInstruct, True: DPO_Instruct
   ```

   Line 101-103: Update RUN_NAME (REVISED NAMING)

   ```python
   RUN_NAME = "DPO_Instruct" if USE_INSTRUCTION else "DPO_NoInstruct"
   HF_REPO = get_model_repo_name(RUN_NAME, precision="bf16")
   ```

   Line 235-239: Replace formatter import and usage

   ```python
   from data_prep.formatters import format_pku_for_dpo, format_pku_for_dpo_instruct

   # Format for DPO (conditional: WITH or WITHOUT instruction)
   formatter = format_pku_for_dpo_instruct if USE_INSTRUCTION else format_pku_for_dpo
   train_dataset = dataset_split['train'].map(
       formatter,
       remove_columns=dataset_split['train'].column_names,
       desc=f"Formatting PKU for DPO ({'WITH' if USE_INSTRUCTION else 'NO'} instruction)"
   )

   val_dataset = dataset_split['test'].map(
       formatter,
       remove_columns=dataset_split['test'].column_names,
       desc=f"Formatting PKU validation for DPO ({'WITH' if USE_INSTRUCTION else 'NO'} instruction)"
   )
   ```

   ---
   📁 CITA SCRIPT (comparative_study/03a_CITA_Baseline/Llama3_BF16.py)

   Line 97-99: Update RUN_NAME (REVISED NAMING)

   ```python
   RUN_NAME = "CITA_Instruct" if USE_INSTRUCTION else "CITA_NoInstruct"
   HF_REPO = get_model_repo_name(RUN_NAME, precision="bf16")
   ```

   Line 109: Fix variable name consistency

   ```python
   USE_INSTRUCTION = False  # Changed from USE_INSTRUCTIONS
   ```

   Line 130-136: REPLACE Trial 2 with Trial 5 hyperparameters

   ```python
   # ===== BEST HYPERPARAMETERS FROM OPTUNA =====
   # Trial 5: 1354 steps, eval_loss=0.2791, margin=6.95, accuracy=89.5% (BEST)
   LAMBDA_KL = 0.000520
   LEARNING_RATE = 6.827978e-06
   BETA = 0.1191
   WEIGHT_DECAY = 0.0091
   WARMUP_RATIO = 0.0749  # Auto-scales: SANITY=101 steps, FULL=101 steps
   ```

   Line 231: Formatter already uses USE_INSTRUCTION (no change needed)

   ---
   📁 MODEL REGISTRY (comparative_study/0c_utils/model_utils.py)

   Line 548-553: Add 3 new entries + rename existing (REVISED NAMING)

   ```python
   MODEL_NAME_MAP = {
       "SFT_NoInstruct": "kapilw25/llama3-8b-pku-SFT-NoInstruct-Baseline-NoInstruct",  # Renamed from SFT_Baseline
       "SFT_Instruct": "kapilw25/llama3-8b-pku-SFT-Instruct-Baseline-NoInstruct",  # NEW

       "DPO_NoInstruct": "kapilw25/llama3-8b-pku-DPO-NoInstruct-SFT-NoInstruct",  # Renamed from DPO_Baseline
       "DPO_Instruct": "kapilw25/llama3-8b-pku-DPO-Instruct-SFT-Instruct",  # NEW

       "CITA_NoInstruct": "kapilw25/llama3-8b-pku-CITA-NoInstruct-DPO-NoInstruct",  # NEW
       "CITA_Instruct": "kapilw25/llama3-8b-pku-CITA-Instruct-DPO-Instruct",  # Existing (was CITA_Baseline)
       "CITA_Adaptive": "kapilw25/llama3-8b-pku-CITA-Instruct-DPO-Instruct",
   }
   ```

   ---
   📁 TOXICITY EVAL (comparative_study/05_evaluation/llm_as_judge/toxicity.py)

   Line 52-69: Expand MODELS dict to 7 variants (REVISED NAMING)

   ```python
   MODELS = {
       "Baseline": {
           "hf_repo": None,
           "display_name": "Baseline (Unaligned)",
           "use_instruction": False,
       },
       "SFT_NoInstruct": {  # Renamed from SFT_Baseline
           "hf_repo": "kapilw25/llama3-8b-pku-SFT-NoInstruct-Baseline-NoInstruct",
           "display_name": "SFT NoInstruct",
           "use_instruction": False,
       },
       "SFT_Instruct": {
           "hf_repo": "kapilw25/llama3-8b-pku-SFT-Instruct-Baseline-NoInstruct",
           "display_name": "SFT Instruct",
           "use_instruction": True,
       },
       "DPO_NoInstruct": {  # Renamed from DPO_Baseline
           "hf_repo": "kapilw25/llama3-8b-pku-DPO-NoInstruct-SFT-NoInstruct",
           "display_name": "DPO NoInstruct",
           "use_instruction": False,
       },
       "DPO_Instruct": {
           "hf_repo": "kapilw25/llama3-8b-pku-DPO-Instruct-SFT-Instruct",
           "display_name": "DPO Instruct",
           "use_instruction": True,
       },
       "CITA_NoInstruct": {  # Renamed from CITA_Baseline
           "hf_repo": "kapilw25/llama3-8b-pku-CITA-NoInstruct-DPO-NoInstruct",
           "display_name": "CITA NoInstruct",
           "use_instruction": False,
       },
       "CITA_Instruct": {
           "hf_repo": "kapilw25/llama3-8b-pku-CITA-Instruct-DPO-Instruct",
           "display_name": "CITA Instruct",
           "use_instruction": True,
       },
   }
   ```

   Line 305-314: Update inference logic

   ```python
   # Format prompts with instruction conditioning based on model config
   formatted = []
   for p, harm_cats in zip(batch_prompts, batch_harm_cats):
       model_config = MODELS[model_key]
       if model_config["use_instruction"]:
           # Models trained WITH instructions: Add system instruction
           instruction = synthesize_system_instruction(harm_cats)
           messages = [
               {"role": "system", "content": instruction},
               {"role": "user", "content": p}
           ]
       else:
           # Models trained WITHOUT instructions: Only user prompt
           messages = [{"role": "user", "content": p}]

       formatted.append(tokenizer.apply_chat_template(
           messages, tokenize=False, add_generation_prompt=True
       ))
   ```

   ---
   ✅ IMPLEMENTATION SUMMARY

   Files Modified: 6
   1. formatters.py - Add 2 new formatters + 2 aliases
   2. SFT script - Add USE_INSTRUCTION toggle + conditional formatter + RUN_NAME="SFT_NoInstruct"
   3. DPO script - Add USE_INSTRUCTION toggle + conditional formatter + RUN_NAME="DPO_NoInstruct"
   4. CITA script - Fix variable name + Trial 5 HPs + RUN_NAME="CITA_NoInstruct"
   5. model_utils.py - Rename 3 keys + add 3 new MODEL_NAME_MAP entries
   6. toxicity.py - Expand to 7 models + rename 3 keys + use_instruction logic

   Naming Convention (Revised):
   - *_NoInstruct = No system instruction (standard training)
   - *_Instruct = With system instruction (CITA-style conditioning)

   Trial 5 Hyperparameters Applied:
   - Best eval_loss: 0.2791 (vs Trial 2's 0.3150)
   - Best margin: 6.95 (vs Trial 2's 4.27)
   - Optimal lambda_kl: 0.000520 (vs Trial 2's 0.000521)

   Ready for Phase 2 Training (9-18 GPU hours):
   1. STAGE 1: SFT_NoInstruct + SFT_Instruct (~1.5h each)
   2. STAGE 2: DPO_NoInstruct + DPO_Instruct (~3.5h each)
   3. STAGE 3: CITA_NoInstruct + CITA_Instruct (~4h each)
   Total: ~9h (parallel stages) or ~18h (sequential)
