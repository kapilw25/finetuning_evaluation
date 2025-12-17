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
| 13  | Negative KL explosion       | 3     | ✅ Fixed            | Policy diverging from reference | 535 |

**Error #12 Details:**
- **Solution:** batch_size 16→8, mini_batch 4→2, max_tokens 256→128
- **Result:** Training progressing (Step 0→80), no OOM, BUT negative KL exploding (-610) → Error #13

**Error #13 Details:**
- **Problem:** Rewards degrading after Step 20 (-3.53→-4.26), KL exploding negatively (0→-610)
- **Root cause:** No gradient clipping → policy diverging too quickly from reference
- **Solution:** Added `max_grad_norm=1.0` (match CITA)
- **Action:** Deleted checkpoint-81 (bad trajectory), will retrain with clipping

