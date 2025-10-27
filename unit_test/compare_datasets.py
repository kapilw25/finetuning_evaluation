#!/usr/bin/env python3
"""
Compare PKU-SafeRLHF vs Vaibhaav/alignment-instructions datasets
Extract top 100 prompts from each and analyze differences
"""

from datasets import load_dataset
import pandas as pd
from collections import Counter

print("="*80)
print("DATASET COMPARISON: PKU-SafeRLHF vs Vaibhaav/alignment-instructions")
print("="*80)

# ============================================================================
# 1. Load PKU-SafeRLHF (Training Split)
# ============================================================================
print("\n1. Loading PKU-SafeRLHF (train split)...")
pku_train = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train")

# Filter for clear safety contrast (same as training code)
pku_filtered = pku_train.filter(
    lambda x: x['is_response_0_safe'] != x['is_response_1_safe']
)

print(f"   Total samples: {len(pku_train):,}")
print(f"   Filtered (clear contrast): {len(pku_filtered):,}")

# Extract prompts and harm categories
pku_prompts = []
pku_harm_cats = []
for ex in pku_filtered.select(range(min(100, len(pku_filtered)))):
    pku_prompts.append(ex['prompt'])

    # Get harm categories from unsafe response
    if not ex['is_response_0_safe']:
        harm_cat = ex['response_0_harm_category']
    else:
        harm_cat = ex['response_1_harm_category']

    categories = [k for k, v in harm_cat.items() if v == True]
    pku_harm_cats.extend(categories)

print(f"   Top 100 prompts extracted")
print(f"   Harm category distribution:")
for cat, count in Counter(pku_harm_cats).most_common():
    print(f"      {cat}: {count}")

# ============================================================================
# 2. Load Vaibhaav/alignment-instructions
# ============================================================================
print("\n" + "="*80)
print("2. Loading Vaibhaav/alignment-instructions (train split)...")
vaibhaav = load_dataset("Vaibhaav/alignment-instructions", split="train")

print(f"   Total samples: {len(vaibhaav):,}")

# Extract first 100 prompts and instructions
vaibhaav_prompts = []
vaibhaav_instructions = []
vaibhaav_prompt_types = []

for ex in vaibhaav.select(range(100)):
    prompt = ex['Prompt'].strip()
    instruction = ex['Instruction generated'].strip()

    vaibhaav_prompts.append(prompt)
    vaibhaav_instructions.append(instruction)

    # Categorize prompt type
    if any(word in prompt.lower() for word in ['hack', 'illegal', 'steal', 'kill', 'harm']):
        vaibhaav_prompt_types.append('harmful')
    else:
        vaibhaav_prompt_types.append('benign')

print(f"   Top 100 prompts extracted")
print(f"   Prompt type distribution:")
for ptype, count in Counter(vaibhaav_prompt_types).items():
    print(f"      {ptype}: {count}")

# ============================================================================
# 3. VAIBHAAV RESPONSE QUALITY CHECK (CRITICAL!)
# ============================================================================
print("\n" + "="*80)
print("3. VAIBHAAV RESPONSE QUALITY CHECK")
print("="*80)
print("   Verifying Accepted vs Rejected responses are actually opposite...")

vaibhaav_accepted = []
vaibhaav_rejected = []
identical_count = 0
length_diff = []

for ex in vaibhaav.select(range(100)):
    accepted = ex['Accepted Response'].strip()
    rejected = ex['Rejected Response'].strip()

    vaibhaav_accepted.append(accepted)
    vaibhaav_rejected.append(rejected)

    # Check if responses are identical
    if accepted == rejected:
        identical_count += 1

    # Track length differences
    length_diff.append(len(accepted) - len(rejected))

print(f"\n   📊 Response Quality Metrics:")
print(f"      Identical responses: {identical_count}/100 ({identical_count}%)")
print(f"      Different responses: {100-identical_count}/100 ({100-identical_count}%)")
print(f"      Avg length diff (Accepted - Rejected): {sum(length_diff)/len(length_diff):.1f} chars")

# Sample comparison - show ENDINGS (where differences are)
print(f"\n   📝 Sample Response Comparisons (Last 150 chars of each):")
for i in range(min(5, len(vaibhaav_prompts))):
    acc = vaibhaav_accepted[i]
    rej = vaibhaav_rejected[i]

    # Find where they diverge
    diverge_idx = -1
    min_len = min(len(acc), len(rej))
    for j in range(min_len):
        if acc[j] != rej[j]:
            diverge_idx = j
            break

    print(f"\n   Sample {i+1} ({vaibhaav_prompt_types[i]}):")
    print(f"      Prompt: {vaibhaav_prompts[i].split('Conversation:')[1][:80] if 'Conversation:' in vaibhaav_prompts[i] else vaibhaav_prompts[i][:80]}...")
    print(f"      Lengths: Accepted={len(acc)}, Rejected={len(rej)}")

    if diverge_idx > 0:
        print(f"      Diverge at char {diverge_idx}")
        print(f"      Accepted[{diverge_idx}:]: {acc[diverge_idx:diverge_idx+80]}...")
        print(f"      Rejected[{diverge_idx}:]: {rej[diverge_idx:diverge_idx+80]}...")
    else:
        print(f"      Accepted ending: ...{acc[-120:]}")
        print(f"      Rejected ending: ...{rej[-120:]}")

# ============================================================================
# 4. Save Sample Comparisons
# ============================================================================
print("\n" + "="*80)
print("4. Saving sample comparisons...")

# PKU samples
pku_df = pd.DataFrame({
    'Dataset': 'PKU-SafeRLHF',
    'Prompt': pku_prompts[:100],
    'Instruction_Type': 'Generic (harm categories)',
    'Sample_Instruction': ['Synthesized from metadata'] * len(pku_prompts[:100])
})

# Vaibhaav samples
vaibhaav_df = pd.DataFrame({
    'Dataset': 'Vaibhaav',
    'Prompt': vaibhaav_prompts,
    'Instruction_Type': 'Natural language (custom)',
    'Sample_Instruction': [inst[:100] + '...' if len(inst) > 100 else inst for inst in vaibhaav_instructions]
})

# Combine and save
comparison_df = pd.concat([pku_df, vaibhaav_df], ignore_index=True)
output_file = "unit_test/dataset_comparison_top100.csv"
comparison_df.to_csv(output_file, index=False)

print(f"   ✅ Saved: {output_file}")

# ============================================================================
# 5. Side-by-Side Comparison (First 5 Samples)
# ============================================================================
print("\n" + "="*80)
print("5. SIDE-BY-SIDE COMPARISON (First 5 Samples)")
print("="*80)

print("\n--- PKU-SafeRLHF Samples ---")
for i in range(5):
    print(f"\nSample {i+1}:")
    print(f"  Prompt: {pku_prompts[i][:80]}...")
    print(f"  Instruction: Generic (synthesized from harm categories)")

print("\n" + "="*80)
print("--- Vaibhaav/alignment-instructions Samples ---")
for i in range(5):
    print(f"\nSample {i+1}:")
    print(f"  Prompt: {vaibhaav_prompts[i][:80]}...")
    print(f"  Instruction: {vaibhaav_instructions[i][:120]}...")

# ============================================================================
# 6. Key Differences Summary
# ============================================================================
print("\n" + "="*80)
print("6. KEY DIFFERENCES SUMMARY")
print("="*80)

print("\n| Aspect | PKU-SafeRLHF | Vaibhaav |")
print("|--------|--------------|----------|")
print(f"| **Train samples** | {len(pku_filtered):,} | {len(vaibhaav):,} |")
print(f"| **Instruction type** | Generic categories | Natural language |")
print(f"| **Instruction diversity** | Low (19 categories) | High (custom per prompt) |")
print(f"| **Harmful prompts** | 100% (filtered) | ~{Counter(vaibhaav_prompt_types)['harmful']}% (first 100) |")
print(f"| **Response quality** | 100% different (filtered) | {100-identical_count}% different |")

print("\n" + "="*80)
print("7. FINAL VERDICT")
print("="*80)
if identical_count > 5:
    print("   ⚠️  WARNING: High duplicate rate in Vaibhaav dataset!")
    print(f"      {identical_count}% of responses are identical (Accepted = Rejected)")
    print("      This will hurt DPO training (needs clear preference signal)")
    print("      Recommendation: Stick with PKU-SafeRLHF for now")
else:
    print("   ✅ Vaibhaav dataset looks good for training!")
    print(f"      Only {identical_count}% duplicates, {100-identical_count}% have clear contrast")
    print("      Proceed with dataset migration plan")

print("\n✅ Analysis complete!")
print(f"   Full comparison saved to: {output_file}")
