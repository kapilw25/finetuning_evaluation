# Distributed Inference for Evaluation

## When to Use Distributed vs Separate Instances

**Key Insight:** Distributed is primarily for **large models that don't fit on single GPU**. For independent jobs, **separate instances are simpler and more efficient**.

---

## When to Use Each Approach

| Scenario | Approach | Why |
|----------|----------|-----|
| **Model > GPU memory (70B+)** | Distributed (tensor/pipeline parallel) | No choice - model must be split |
| **Independent evals/trains** | Separate instances | Simpler, no coordination overhead, fault-tolerant |

---

## Llama-3.1-8B Case

| Resource | Requirement |
|----------|-------------|
| Model size | ~16GB (BF16) |
| A100-40GB | Fits easily with 24GB headroom |
| A100-80GB | Fits with 64GB headroom |

**Verdict:** No need for distributed. Run separate instances.

---

## Why Separate Instances > Distributed for Independent Jobs

| Factor | Distributed (4 nodes tightly coupled) | Separate Instances (4 independent) |
|--------|---------------------------------------|-----------------------------------|
| **Failure handling** | 1 node fails → ALL stop | 1 node fails → 3 continue |
| **Idle time** | All wait for slowest | Each finishes independently |
| **Setup complexity** | High (NCCL, networking, sync) | None |
| **Debugging** | Hard (distributed logs) | Easy (single process) |
| **Cost efficiency** | Pay for idle nodes waiting | Each node utilized 100% |

---

## Recommendation for Evaluations

```
❌ DON'T: 4-node distributed cluster for 5 eval scripts
✅ DO: Run each eval on separate instance independently
```

### Instance Assignment Example

```bash
# Instance 1
python comparative_study/05_evaluation/truthfulqa/evaluation.py --mode full

# Instance 2
python comparative_study/05_evaluation/conditional_safety/evaluation.py --mode full

# Instance 3
python comparative_study/05_evaluation/length_control/evaluation.py --mode full
python comparative_study/05_evaluation/AQI/evaluation.py --mode full

# Instance 4
python comparative_study/05_evaluation/isd/evaluation.py --mode full
```

Each instance runs independently. If one fails, others continue. No wasted idle time.

---

## Data Sharding (Optional)

The `distributed_eval.sh` script in this directory is for **data-sharding** - splitting a SINGLE large evaluation across multiple nodes for speed. This is different from model-parallelism.

### When Data Sharding Helps

- Very large evaluation datasets (10K+ samples)
- Time-critical evaluations
- Each node has identical GPU (no stragglers)

### Usage

```bash
# On 4 separate nodes (run simultaneously):
./distributed_eval.sh --eval truthfulqa --nodes 4 --node-id 0 --mode full
./distributed_eval.sh --eval truthfulqa --nodes 4 --node-id 1 --mode full
./distributed_eval.sh --eval truthfulqa --nodes 4 --node-id 2 --mode full
./distributed_eval.sh --eval truthfulqa --nodes 4 --node-id 3 --mode full

# After ALL nodes complete, merge on any node:
./distributed_eval.sh --eval truthfulqa --nodes 4 --merge
```

### Supported Evaluations

- `truthfulqa` - TruthfulQA (817 questions)
- `conditional_safety` - Safety adaptation (500 prompts)
- `length_control` - Length following
- `aqi` - Answer Quality Index
- `isd` - Instruction-State Dependence

---

## Files

```
distributed_inference/
├── README.md              # This file
├── distributed_eval.sh    # Data sharding orchestration script
└── utils/
    ├── __init__.py
    └── sharding.py        # Dataset sharding utilities
```
