# Alignment Controllability Evolution

## System Design Diagram (Horizontal: Left → Right)

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                              ALIGNMENT CONTROLLABILITY EVOLUTION                                                                                                               │
└────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────┐             ┌────────────────────────────────────────────┐             ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│  LEVEL 0: STATIC ALIGNMENT                │             │  LEVEL 1: ECLIPTICA + CITA                 │             │  LEVEL 2: SWISS KNIFE                                                                       │
│  (DPO / RLHF / GRPO)                      │             │  (Instruction-Conditioned Switching)       │             │  (Externalized Decode-Time Alignment)                                                       │
│  ══════════════════════                   │             │  ═══════════════════════════════════       │             │  ══════════════════════════════════════                                                     │
│                                           │             │                                            │             │                                                                                             │
│                                           │             │  ┌───────────┐                             │             │                                                                                             │
│  ┌─────────┐   ┌──────────────────┐       │             │  │  Instr I  │──┐                          │             │  ┌─────────┐   ┌───────────────────────────────────────────────────────────────────────┐   │
│  │Prompt X │──▶│ 🔒 FROZEN CKPT   │──▶ Y  │             │  │ "safety_  │  │  ┌──────────────────┐    │             │  │Prompt X │──▶│  SPECULATIVE DECODING PIPELINE                                        │   │
│  └─────────┘   │                  │       │             │  │  first"   │  ├─▶│ 🔄 SWITCHABLE    │──▶Y|I           │  └─────────┘   │                                                                       │   │
│                │ Single hardcoded │       │             │  └───────────┘  │  │    BACKBONE      │    │             │                │  ┌─────────────┐   ┌─────────────────────┐   ┌──────────────────┐   │   │
│                │ policy:          │       │             │                 │  │                  │    │             │                │  │ DRAFT MODEL │──▶│ TOURNAMENT SAMPLING │──▶│   Response Y     │   │   │
│                │ • 1 refusal mode │       │             │  ┌───────────┐  │  │ π_θ(·|I,X) via  │    │             │                │  │ (Fast)      │   │ AUDITOR (TSA)       │   └──────────────────┘   │   │
│                │ • 1 verbosity    │       │             │  │ Prompt X  │──┘  │ CITA:            │    │             │                │  │             │   │                     │            ▲            │   │
│                │ • 1 safety level │       │             │  └───────────┘     │                  │    │             │                │  │ K candidates│   │ Bracket selection   │            │            │   │
│                └──────────────────┘       │             │                    │ L_pref + λ·KL   │    │             │                │  └─────────────┘   └─────────────────────┘            │            │   │
│                                           │             │                    │                  │    │             │                │                             ▲                        │            │   │
│                                           │             │                    │ KL anchor keeps  │    │             │                │  ┌───────────────────────────┴────────────────────────┘            │   │
│                                           │             │                    │ regimes stable   │    │             │                │  │           🔌 PLUGGABLE BLADES (Hot-Swappable)                    │   │
│                                           │             │                    └──────────────────┘    │             │                │  │  ┌─────────┬──────────┬───────────┬─────────┐                   │   │
│                                           │             │                                            │             │                │  │  │ Safety  │ Helpful  │ Harmless  │  Style  │                   │   │
│                                           │             │                                            │             │                │  │  │  Blade  │  Blade   │   Blade   │  Blade  │                   │   │
│                                           │             │                                            │             │                │  │  └─────────┴──────────┴───────────┴─────────┘                   │   │
│                                           │             │                                            │             │                │  └──────────────────────────────────────────────────────────────────┘   │
│                                           │             │                                            │             │                └───────────────────────────────────────────────────────────────────────┘   │
├───────────────────────────────────────────┤             ├────────────────────────────────────────────┤             ├─────────────────────────────────────────────────────────────────────────────────────────────┤
│ ❌ Need separate ckpt per policy          │             │ ✅ One ckpt, many policies via I           │             │ ✅ Alignment fully externalized                                                             │
│ ❌ Expensive, unscalable                  │════════════▶│ ❌ Backbone internalizes ALL regimes       │════════════▶│ ✅ Hot-swap auditors post-deployment                                                        │
│ ❌ No runtime policy control              │   solves    │ ❌ Cannot hot-update post-deployment       │   solves    │ ✅ Train small auditors independently                                                       │
└───────────────────────────────────────────┘             └────────────────────────────────────────────┘             └─────────────────────────────────────────────────────────────────────────────────────────────┘
              │                                                         │                                                          │
              ▼                                                         ▼                                                          ▼
┌───────────────────────────────────────────┐             ┌────────────────────────────────────────────┐             ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ CONTROL: None (baked in weights)          │             │ CONTROL: Natural language instruction I    │             │ CONTROL: Swap auditor "blade" module                                                        │
│ UPDATE:  Retrain entire model             │             │ UPDATE:  Retrain backbone with new regimes │             │ UPDATE:  Train/swap small auditor only                                                      │
│ COST:    $$$$ (full training per policy)  │             │ COST:    $$$ (one training, multiple I)    │             │ COST:    $ (small auditor training)                                                         │
└───────────────────────────────────────────┘             └────────────────────────────────────────────┘             └─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Compact Summary View

```
STATIC ALIGNMENT ═══════════════════════════▶ ECLIPTICA/CITA ═══════════════════════════▶ SWISS KNIFE
(Checkpoint-Bound)                            (Instruction-Conditioned)                    (Decode-Time Modular)

┌────────────────────────────┐     ┌────────────────────────────┐     ┌──────────────────────────────────────────┐
│                            │     │                            │     │                                          │
│  X ──▶ [🔒 FROZEN] ──▶ Y   │     │  I ─┬▶ [🔄 BACKBONE] ──▶ Y │     │  X ──▶ [DRAFT] ──▶ [TSA] ──▶ Y           │
│                            │     │  X ─┘   π_θ(·|I,X)         │     │                      ▲                   │
│  One policy per ckpt       │     │                            │     │           ┌───┬───┬───┼───┐              │
│                            │     │  L_CITA + KL anchor        │     │           │ S │ H │ H │ St│ ◀─ Blades   │
│                            │     │                            │     │           └───┴───┴───┴───┘              │
└────────────────────────────┘     └────────────────────────────┘     └──────────────────────────────────────────┘
          │                                    │                                       │
          ▼                                    ▼                                       ▼
    ❌ Separate ckpt/policy             ✅ Multi-policy/ckpt                   ✅ Externalized alignment
    ❌ No runtime update                ❌ Regimes interfere                   ✅ Hot-swap auditors
                                        ❌ No hot-update                       ✅ Cheap auditor training
```

---

## Side-by-Side Comparison Table

| Dimension               | LEVEL 0: Static          | LEVEL 1: ECLIPTICA/CITA        | LEVEL 2: Swiss Knife            |
|-------------------------|--------------------------|--------------------------------|---------------------------------|
| Where alignment lives   | Backbone weights         | Backbone weights               | External auditor heads          |
| Switching mechanism     | None (retrain)           | Change instruction I           | Swap auditor "blade"            |
| Training scope          | Full model per policy    | Full backbone (8B params)      | Small auditors (~100M)          |
| Post-deployment update  | Retrain everything       | Retrain with new regimes       | Hot-swap auditor module         |
| Regime interference     | N/A (single regime)      | Possible (KL mitigates)        | Fully decoupled                 |
| Inference overhead      | None                     | None (just add instruction)    | K candidates + tournaments      |
| Cost                    | $$$$                     | $$$                            | $                               |
