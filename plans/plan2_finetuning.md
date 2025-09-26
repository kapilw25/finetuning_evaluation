# Plan 2: Finetuning Configuration Optimization

## Problem Analysis - Current QLoRA Training Failure

### Root Cause: Severe Under-training
**Training Results:**
- Initial loss: 2.7335
- Final loss: 2.7150
- Improvement: Only 0.0185 (0.68%) - **NO MEANINGFUL LEARNING**

### Critical Issues Identified:

#### 1. **Insufficient Training Coverage**
- **Dataset size**: 45,000 samples
- **Max steps**: 500 steps only
- **Effective batch size**: 1 × 8 = 8 samples per update
- **Coverage**: 500 × 8 = 4,000 samples H **8.9% of dataset seen**
- **Problem**: Model barely saw the training data

#### 2. **Learning Rate Decay Too Aggressive**
- **Initial LR**: 5e-5
- **Final LR**: ~5e-10 (practically zero)
- **Issue**: LR became ineffective after ~200 steps with cosine decay
- **Gradient norm**: 0.0 throughout training (major red flag)

#### 3. **Quantization + Small Batch = Unstable Training**
- 4-bit NF4 double quantization
- Batch size = 1 per device
- **Result**: Gradients too noisy/unstable for effective learning

#### 4. **Evaluation Metrics Stagnant**
- Eval loss: 2.9310 at steps 150, 300, 450 (identical)
- Token accuracy: 45.2% ’ 46.0% (minimal improvement)
- **Conclusion**: No real learning occurred

## Solution: Optimized Training Configuration

### Approach 1: Conservative Fix (Recommended)
```python
training_args = SFTConfig(
    per_device_train_batch_size=2,           # Double batch size
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=16,          # Effective batch = 32
    learning_rate=1e-4,                      # Higher LR
    max_steps=2000,                          # 4x more steps
    warmup_steps=200,                        # Proper warmup
    lr_scheduler_type="cosine",              # Slower decay
    logging_steps=25,
    eval_steps=500,                          # Every 25% progress
    save_steps=500,
    eval_strategy="steps",
    save_strategy="steps",
    max_grad_norm=1.0,                       # Fix gradient clipping
)
```

**Coverage**: 2000 steps × 32 = 64,000 samples H **142% of dataset** (1.4 epochs)

### Approach 2: Aggressive Fix
```python
training_args = SFTConfig(
    per_device_train_batch_size=4,           # Larger batch
    gradient_accumulation_steps=16,          # Effective batch = 64
    learning_rate=2e-4,                      # Higher LR
    max_steps=5000,                          # Full training
    warmup_steps=500,                        # Longer warmup
    weight_decay=0.01,                       # Add regularization
    max_grad_norm=1.0,
)
```

**Coverage**: 5000 steps × 64 = 320,000 samples H **711% of dataset** (7+ epochs)

### Approach 3: Quantization Alternative
```python
# Switch to 8-bit quantization for stability
BitsAndBytesConfig(
    load_in_8bit=True,                       # More stable than 4-bit
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
)

# LoRA config for 8-bit
lora_config = LoraConfig(
    r=32,                                    # Higher rank
    lora_alpha=64,                           # 2:1 ratio
    target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # Remove embed_tokens/lm_head
    lora_dropout=0.1,
    bias="none",
    use_rslora=True,
)
```

## Implementation Priority

### Phase 1: Quick Fix (Conservative Approach)
1. **Update `m03a_qlora_training.py`** with Approach 1 config
2. **Test with 2000 steps** to validate learning occurs
3. **Monitor**: Loss should drop significantly (>0.2) in first 500 steps

### Phase 2: Comprehensive Fix
1. **Implement Approach 2** if Phase 1 shows improvement
2. **A/B test** 4-bit vs 8-bit quantization
3. **Dataset optimization**: Check for duplicates, quality filtering

### Phase 3: Advanced Optimization
1. **Learning rate scheduling**: Implement custom scheduler
2. **Dynamic batch sizing**: Start small, increase gradually
3. **Gradient analysis**: Add gradient norm logging and clipping optimization

## Success Metrics

### Minimum Success Criteria:
- **Loss drop**: >0.5 from initial (2.7 ’ 2.2 or better)
- **Eval loss improvement**: >0.2 improvement
- **Token accuracy**: >5% absolute improvement
- **Gradient norms**: >0.0 (actual learning happening)

### Optimal Success Criteria:
- **Loss drop**: >1.0 from initial (2.7 ’ 1.7 or better)
- **Perplexity**: <6.0 on validation set
- **Generation quality**: Coherent responses to test prompts
- **Training stability**: Consistent loss reduction over time

## Risk Mitigation

### Memory Management:
- **Gradient checkpointing**: Enable to reduce VRAM
- **Flash Attention**: Use for long sequences
- **Mixed precision**: FP16 training

### Training Stability:
- **Early stopping**: Stop if eval loss stops improving
- **Checkpoint frequency**: Save every 500 steps
- **Gradient monitoring**: Alert if grad_norm = 0.0

### Quality Assurance:
- **Validation prompts**: Test generation quality during training
- **Loss curves**: Plot and analyze training/eval loss
- **Parameter analysis**: Monitor LoRA weight magnitudes

## Expected Timeline
- **Phase 1**: 2-3 hours training time
- **Phase 2**: 8-12 hours training time
- **Phase 3**: 1-2 days optimization cycle

## Next Steps
1. Implement Approach 1 configuration in `src/m03a_qlora_training.py`
2. Run training with 2000 steps
3. Analyze results and proceed to Phase 2 if successful
4. Document findings and update evaluation framework