  Goal: SFT < SFT < DPO, GRPO << CITA in @Overleaf_draft/figures/evaluation/combined_plots/radar.png
  
  
  
  ---
  Summary Table

  | #   | Error                     | Session | Status              | Impact                | Line    |
  |-----|---------------------------|---------|---------------------|-----------------------|---------|
  | 1   | Batch size constraint     | 1       | ✅ Fixed            | Training can start    | 489-496 |
  | 2   | Variable-length sequences | 1       | ✅ Fixed            | Batching works        | 539-546 |
  | 3   | Query column tensor       | 1       | ✅ Fixed            | Data collator works   | 245     |
  | 4   | Query tensor shape        | 1       | ✅ Fixed            | Generation works      | 591-592 |
  | 5   | Negative KL divergence    | 1       | ⚠️ Non-critical     | CITA has same         | 525-532 |
  | 6   | PEFT detection broken     | 1       | ❌ Led to Session 2 | 96% batches skipped   | 434-439 |
  | 7   | Missing checkpoints       | 2       | ✅ Fixed            | Can resume training   | 687-694 |
  | 8   | Low GPU utilization       | 2       | ✅ Fixed            | 4x faster (led to #9) | 489-496 |
  | 9   | Adapter loading failure   | 3       | ✅ Fixed            | Gradients enabled     | 404-425 |
  | 10  | OOM from optimization     | 3       | ✅ Fixed            | Balanced config       | 493-495 |
  | 11  | TypeError set_adapter     | 3       | ✅ Fixed            | Single adapter active | 430     |
| 12  | OOM from explicit ref_model | 3     | ✅ Fixed            | Dual-model needs less memory | 516-518, 547 |
| 13  | Negative KL explosion       | 3     | ⚠️ Partial          | Gradient clipping helped but not enough | 535 |
| 14  | Wrong ref_model creation    | 4     | ✅ Fixed            | Use TRL create_reference_model() | 412-418 |
| 15  | KL penalty mode default     | 4     | ✅ Fixed            | Use kl_penalty='full' | 504 |

**Error #12 Details:**
- **Solution:** batch_size 16→8, mini_batch 4→2, max_tokens 256→128
- **Result:** Training progressing (Step 0→80), no OOM, BUT negative KL exploding (-610) → Error #13

**Error #13 Details:**
- **Problem:** Rewards degrading after Step 20 (-3.53→-4.26), KL exploding negatively (0→-610)
- **Root cause:** No gradient clipping → policy diverging too quickly from reference
- **Solution:** Added `max_grad_norm=1.0` (match CITA)
- **Result:** Helped but still seeing negative KL in Instruct run → Led to Error #14

**Error #14 Details:**
- **Problem:** Negative KL divergence persisting despite gradient clipping
  - NoInstruct: KL=0.0000 (exactly zero - suspicious)
  - Instruct: KL=-2.38 → -8.56 (exploding negative)
- **Root cause:** Wrong reference model creation method
  - OLD: `copy.deepcopy(model.pretrained_model)` + `ValueHeadModel.from_pretrained()`
  - This lost wrapper context and created malformed reference
- **Solution:** Use TRL's official `create_reference_model(model)` function
  - Creates proper frozen copy with shared layers for memory efficiency
  - See: https://huggingface.co/docs/trl/en/models#trl.create_reference_model
- **Code change:** Lines 412-418 in Llama3_BF16.py

**Error #15 Details:**
- **Problem:** KL still going negative despite create_reference_model fix
  - Step 10: KL=4.83 (healthy initially)
  - Then: KL=-1.81 → -4.37 (exploding negative again)
- **Root cause:** Default `kl_penalty='kl'` uses approximate KL on target tokens only
  - Approximate KL can be negative when `log_p_active < log_p_ref`
  - See: https://github.com/huggingface/trl/issues/1017
- **Solution:** Set `kl_penalty='full'` in PPOConfig
  - Full KL is mathematically non-negative
- **Code change:** Line 504 in Llama3_BF16.py

