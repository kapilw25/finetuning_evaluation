# How to Use Anthropic/hh-rlhf for SFT, DPO, and CITA Training

## Dataset Structure

**Anthropic/hh-rlhf** contains multi-turn conversations:
- `chosen`: Helpful & harmless response
- `rejected`: Less helpful or harmful response

Format: `\n\nHuman: ... \n\nAssistant: ...`

---

## Method 1: SFT (Supervised Fine-Tuning)

**Use only the "chosen" responses**

```python
from datasets import load_dataset

dataset = load_dataset("Anthropic/hh-rlhf", split="train")

def format_sft(example):
    """
    SFT only needs: Input → Output pairs
    Use the CHOSEN response (safe/helpful)
    """
    # The "chosen" field already contains the full conversation
    # For chat models, keep it as-is
    # For instruction models, you might want to parse it

    conversation = example['chosen']

    # Parse into messages format (for Llama-3 chat template)
    messages = []
    turns = conversation.split('\n\nHuman:')[1:]  # Skip empty first element

    for turn in turns:
        if '\n\nAssistant:' in turn:
            human_text, assistant_text = turn.split('\n\nAssistant:')
            messages.append({"role": "user", "content": human_text.strip()})
            messages.append({"role": "assistant", "content": assistant_text.strip()})
        else:
            # Incomplete turn (just human, no assistant response)
            messages.append({"role": "user", "content": turn.strip()})

    return {"messages": messages}

sft_dataset = dataset.map(format_sft, remove_columns=dataset.column_names)
```

**Training:**
```python
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=sft_dataset,
    tokenizer=tokenizer,
    # SFT will use tokenizer.apply_chat_template() automatically
)
```

---

## Method 2: DPO (Direct Preference Optimization)

**Use both "chosen" and "rejected"**

```python
def format_dpo(example):
    """
    DPO needs: Prompt + Chosen + Rejected

    The hh-rlhf dataset is ALREADY in the right format!
    Just parse into messages.
    """
    def parse_conversation(conv_text):
        """Parse conversation into messages list"""
        messages = []
        turns = conv_text.split('\n\nHuman:')[1:]

        for turn in turns:
            if '\n\nAssistant:' in turn:
                human_text, assistant_text = turn.split('\n\nAssistant:')
                messages.append({"role": "user", "content": human_text.strip()})
                messages.append({"role": "assistant", "content": assistant_text.strip()})
            else:
                messages.append({"role": "user", "content": turn.strip()})

        return messages

    chosen_messages = parse_conversation(example['chosen'])
    rejected_messages = parse_conversation(example['rejected'])

    return {
        "chosen": chosen_messages,
        "rejected": rejected_messages
    }

dpo_dataset = dataset.map(format_dpo, remove_columns=dataset.column_names)
```

**Training:**
```python
from trl import DPOTrainer

trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=DPOConfig(beta=0.1, ...),
    train_dataset=dpo_dataset,
    processing_class=tokenizer,
)
```

---

## Method 3: CITA (Contrastive Instruction-Tuned Alignment)

**Use both "chosen" and "rejected" WITH explicit system instruction**

⚠️ **KEY DIFFERENCE:** CITA requires an **explicit alignment instruction** in the system role.

### Problem: HH-RLHF doesn't have explicit instructions!

**Solution:** Extract or synthesize the alignment instruction from the conversation context.

```python
def format_cita(example):
    """
    CITA needs: System Instruction + User Prompt + Chosen + Rejected

    Strategy: Add a SYSTEM message with alignment instructions
    """
    def parse_conversation_with_system(conv_text, instruction):
        """Parse and add system instruction"""
        messages = []
        turns = conv_text.split('\n\nHuman:')[1:]

        for i, turn in enumerate(turns):
            if '\n\nAssistant:' in turn:
                human_text, assistant_text = turn.split('\n\nAssistant:')

                # Add system instruction ONLY for first turn
                if i == 0:
                    messages.append({"role": "system", "content": instruction})

                messages.append({"role": "user", "content": human_text.strip()})
                messages.append({"role": "assistant", "content": assistant_text.strip()})
            else:
                if i == 0:
                    messages.append({"role": "system", "content": instruction})
                messages.append({"role": "user", "content": turn.strip()})

        return messages

    # OPTION 1: Use a generic alignment instruction
    instruction = "You are a helpful, harmless, and honest AI assistant. Always prioritize user safety and refuse harmful requests."

    # OPTION 2: Infer instruction from first human turn (more sophisticated)
    # You could analyze the content and generate context-specific instructions
    # Example: If conversation is about illegal activity → "Refuse illegal requests"

    chosen_messages = parse_conversation_with_system(example['chosen'], instruction)
    rejected_messages = parse_conversation_with_system(example['rejected'], instruction)

    return {
        "chosen": chosen_messages,
        "rejected": rejected_messages
    }

cita_dataset = dataset.map(format_cita, remove_columns=dataset.column_names)
```

**Training:**
```python
from cita_trainer import CITATrainer  # Your custom CITA trainer

trainer = CITATrainer(
    model=model,
    ref_model=ref_model,
    args=DPOConfig(beta=0.1, ...),
    train_dataset=cita_dataset,
    processing_class=tokenizer,
    lambda_kl=0.01,
)
```

---

## Alternative: Use PKU-SafeRLHF Instead

**Better choice for CITA** because it has **explicit safety preferences**:

```python
dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF")
# This dataset has:
# - prompt
# - response_0 (safer)
# - response_1 (less safe)
# - safer_response_id (0 or 1)
# - is_response_0_safe, is_response_1_safe (boolean flags)
```

### Format for CITA with PKU-SafeRLHF:

```python
def format_pku_for_cita(example):
    """
    PKU-SafeRLHF → CITA format
    This dataset has explicit safety labels!
    """
    # Determine which is safer
    safer_idx = example['safer_response_id']

    if safer_idx == 0:
        safe_response = example['response_0']
        unsafe_response = example['response_1']
    else:
        safe_response = example['response_1']
        unsafe_response = example['response_0']

    # Create instruction based on safety type
    instruction = "You are a helpful and safe AI assistant. Always prioritize user safety and refuse harmful, illegal, or unethical requests."

    chosen_messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": example['prompt']},
        {"role": "assistant", "content": safe_response}
    ]

    rejected_messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": example['prompt']},
        {"role": "assistant", "content": unsafe_response}
    ]

    return {
        "chosen": chosen_messages,
        "rejected": rejected_messages
    }
```

---

## Summary Table

| Method | Data Needed | Best Dataset | System Instruction? |
|--------|-------------|--------------|---------------------|
| **SFT** | Input + Safe Output | `Anthropic/hh-rlhf` (chosen only) | ❌ Optional |
| **DPO** | Prompt + Chosen + Rejected | `Anthropic/hh-rlhf` | ❌ No |
| **CITA** | **System Instruction** + Prompt + Chosen + Rejected | `PKU-Alignment/PKU-SafeRLHF` | ✅ **REQUIRED** |

---

## Key Takeaway for CITA

**The main innovation of CITA is explicit instruction conditioning:**

- **DPO:** `P(response | user_prompt)`
- **CITA:** `P(response | system_instruction, user_prompt)`

This means CITA can adapt its behavior based on different alignment directives (e.g., "be helpful" vs "prioritize safety").

Without explicit system instructions in your dataset, CITA reduces to DPO!
